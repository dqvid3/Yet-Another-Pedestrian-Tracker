import torch
import torchvision.transforms as T
from torchvision.ops import RoIAlign

torch.set_grad_enabled(False)

# standard PyTorch mean-std input image normalization
transform_ = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_detr():
    detr = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    detr.eval()
    return detr


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(model, im, transform=transform_, threshold_confidence=0.5):
    img = transform(im).unsqueeze(0)
    assert img.shape[-2] <= 1600 and img.shape[
        -1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    conv_features = []
    hook = model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    )
    outputs = model(img)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold_confidence
    keep &= probas.argmax(-1) == 1  # class 1 is person
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    hook.remove()

    conv_features = conv_features[0]
    features = conv_features['0'].tensors

    return probas[keep], bboxes_scaled, features


def extract_roi_features(conv_features, bboxes, output_size=(7, 7)):
    # Aggiungi gli indici dell'immagine alle bounding box
    # (necessario per roi_align)
    num_boxes = bboxes.shape[0]
    img_indices = torch.zeros((num_boxes, 1), dtype=torch.float32)
    rois = torch.cat([img_indices, bboxes], dim=1)
    detr_scale_factor = 1 / 32
    roi_align = RoIAlign(output_size, detr_scale_factor, -1)

    # Estrai le caratteristiche delle regioni di interesse
    roi_features = roi_align(conv_features, rois)
    # Ridimensiona le caratteristiche in un vettore unidimensionale
    roi_features = roi_features.view(roi_features.size(0), -1)

    return roi_features
