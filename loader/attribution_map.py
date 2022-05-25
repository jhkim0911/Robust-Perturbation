import h5py
from util.ops import *

def update_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def robust(data, model, device, class_num, rp_dir, i_idx, iter_num, visual=False):
    rp_h5dir = rp_dir + '/h5'
    check_dir(rp_h5dir)

    input_img, annot, c_shape, img_root = data
    m = torch.nn.Sigmoid()
    max_iterations = iter_num

    labels = []
    bbox = []
    img_root = img_root[0]

    # Generates bounding box information to match specific format
    for i in range(len(annot)):
        labels.append(annot[i][0].long())
        bbox.append(torch.cat(annot[i][1:], dim=0))

    target = torch.zeros(len(torch.cat(labels, dim=0)), class_num).scatter_(1, torch.cat(labels, dim=0).unsqueeze(1), 1.)
    bbox_info = torch.cat(bbox, dim=0).view(len(annot), -1)
    ori_shape = torch.cat(c_shape, dim=0)

    mat = ori_shape.double().repeat(len(annot), 2)
    mat = mat * bbox_info
    bnd_box = torch.cat((mat[:, :2] - mat[:, 2:] / 2, mat[:, 2:]), dim=-1).round().float()

    idx = (target != 0).nonzero()

    model = model.to(device)
    input_img = input_img.to(device)

    logits = model(Variable(input_img).to(device))

    pred = m(logits)
    pred_idx = (pred >= 0.5).nonzero()

    target_idx = torch.unique(idx[:, 1]).squeeze().to(device)

    # Generating attribution maps only for the correctly predicted labels (Multi-label prediction).
    if torch.equal(target_idx, pred_idx[:, 1].squeeze()):
        for v_idx in range(target_idx.numel()):
            original_img = cv2.imread(img_root, 1)
            original_img = cv2.resize(original_img, (300, 300))
            img = np.float32(original_img) / 255
            blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
            blurred_img2 = np.float32(cv2.medianBlur(original_img, 11)) / 255
            mask_init = np.zeros((300, 300), dtype=np.float32)
            blurred_img = preprocess_image((blurred_img1 + blurred_img2) / 2, device)

            mask = numpy_to_torch(mask_init, device)
            optimizer = torch.optim.Adam([mask], lr=0.1)

            if target_idx.numel() == 1:
                target_idx = target_idx.unsqueeze(0)

            for i in range(max_iterations):
                upsampled_mask = mask

                upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))

                # Use the mask to perturbated the input image.
                perturbated_input = input_img.mul(upsampled_mask) + blurred_img.mul(1 - upsampled_mask)
                preserved_input = input_img.mul(1 - upsampled_mask) + blurred_img.mul(upsampled_mask)

                noise = np.zeros((300, 300, 3), dtype=np.float32)
                cv2.randn(noise, 0, 0.2)
                noise = numpy_to_torch(noise, device=device)
                noise2 = np.zeros((300, 300, 3), dtype=np.float32)
                cv2.randn(noise2, 0, 0.2)
                noise2 = numpy_to_torch(noise2, device=device)

                perturbated_input_ = perturbated_input + noise
                preserved_input_ = preserved_input + noise2

                prt_logits = model(perturbated_input_)
                psv_logits = model(preserved_input_)

                prt_outputs = m(prt_logits)
                psv_output = m(psv_logits)

                cat_idx = target_idx[v_idx].item()

                prt_pred = prt_outputs.clone()[:, cat_idx].unsqueeze(0)
                ori_pred = pred.clone()[:, cat_idx].unsqueeze(0)
                psv_prd = psv_output[:, cat_idx].unsqueeze(0)

                r_outputs_m = torch.cat([prt_outputs.clone()[:, :cat_idx], prt_outputs.clone()[:, cat_idx + 1:]], dim=-1)
                r_pred = torch.cat([pred.clone()[:, :cat_idx], pred.clone()[:, cat_idx + 1:]], dim=-1)
                p_outputs_m = torch.cat([psv_output[:, :cat_idx], psv_output[:, cat_idx + 1:]], dim=-1)

                mask_loss = 5. * torch.mean(abs(1 - mask))
                tv_loss = 20 * tv_norm(mask, 2)

                #perturb_loss
                target_loss = bounded_kl(ori_pred, prt_pred)
                ntarget_loss = bounded_kl(r_pred, r_outputs_m)

                #r-perturb loss
                ptarget_loss = bounded_kl(ori_pred, psv_prd)
                pnt_loss = (torch.log2(torch.abs(p_outputs_m) + 1)).mean()

                loss = -target_loss + ntarget_loss + mask_loss + ptarget_loss + pnt_loss + tv_loss

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # torch.nn.utils.clip_grad_norm(mask, 0.5)
                optimizer.step()
                # Optional: clamping seems to give better results
                mask.data.clamp_(0, 1)

            coutputs = m(model(perturbated_input))
            cpoutputs = m(model(preserved_input))

            attr_map = (1-mask).squeeze().cpu().detach().numpy()

            matched_box = box_matching(idx, target_idx[v_idx], bnd_box)

            # Create h5py files. For the various experiments, we first saved attribution info, then reused them.
            if (attr_map is not None) == True:
                with h5py.File(rp_h5dir + '/VOC' + data[3][0][60:66] + '_%d_%d.h5' %(i_idx, v_idx), 'w') as h5f:
                    h5f.create_dataset('directory', data=data[3][0][60:])
                    h5f.create_dataset('attr_map', data=attr_map)
                    h5f.create_dataset('bbox', data=matched_box)
                    h5f.create_dataset('pred', data=pred.squeeze().cpu().detach().numpy())
                    h5f.create_dataset('prt', data=coutputs.squeeze().cpu().detach().numpy())
                    h5f.create_dataset('psv', data=cpoutputs.squeeze().cpu().detach().numpy())
                    h5f.create_dataset('class', data=target_idx[v_idx].item())
                    h5f.flush()

            if visual == True:
                bg_img = visualization(attr_map, img_root, matched_box, pred_idx, idx)
                bg_img.save(rp_dir + '/' + 'rp%3d_%d.png' % (i_idx, v_idx))
                print(('[*] file: rp%3d_%d.png saved\n' % (i_idx, v_idx)))
                bg_img.close()

        return attr_map
