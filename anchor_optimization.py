import torch
import pandas as pd
import numpy as np
import scipy.optimize
import ast


def calculate_config(values, ratio_count, sizes=[32, 64, 128, 256, 512], strides=[8, 16, 32, 64, 128]):
    split_point = int((ratio_count - 1) / 2)

    ratios = [1]
    for i in range(split_point):
        ratios.append(values[i])
        ratios.append(1 / values[i])

    scales = values[split_point:]

    return sizes, strides, torch.tensor(ratios), torch.tensor(scales)


def generate_anchors(base_size, ratios, scales):
    num_anchors = len(ratios) * len(scales)
    anchors = torch.zeros((num_anchors, 4))

    for i, ratio in enumerate(ratios):
        for j, scale in enumerate(scales):
            width = base_size * scale * ratio ** 0.5
            height = base_size * scale / ratio ** 0.5
            anchors[i * len(scales) + j, :] = torch.tensor([0, 0, width, height])

    return anchors


def base_anchors_for_shape(pyramid_levels=None, anchor_params=None):
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    sizes, strides, ratios, scales = anchor_params

    all_anchors = []
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=sizes[idx],
            ratios=ratios,
            scales=scales
        )
        all_anchors.append(anchors)

    return torch.cat(all_anchors, dim=0)


def compute_overlap(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = torch.zeros((N, K), dtype=torch.float64)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                torch.min(boxes[n, 2], query_boxes[k, 2]) -
                torch.max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    torch.min(boxes[n, 3], query_boxes[k, 3]) -
                    torch.max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def load_annotations(data, include_stride=False):
    entries = []
    image_shape = [768, 1024]

    for index, row in data.iterrows():
        bboxes = ast.literal_eval(row['bbox'])

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)

            if include_stride:
                entries.append([x1, y1, x2, y2])
            else:
                width = x2 - x1
                height = y2 - y1
                entries.append([-width / 2, -height / 2, width / 2, height / 2])

    return torch.tensor(entries, dtype=torch.float64), image_shape


def average_overlap(values, entries, image_shape, mode='avg', ratio_count=3, include_stride=False, sizes=[32, 64, 128, 256, 512], strides=[8, 16, 32, 64, 128], verbose=True):
    sizes, strides, ratios, scales = calculate_config(values, ratio_count, sizes, strides)

    if include_stride:
        raise NotImplementedError
    else:
        anchors = base_anchors_for_shape(anchor_params=(sizes, strides, ratios, scales))

    overlap = compute_overlap(entries, anchors)
    max_overlap = torch.max(overlap, dim=1)[0]

    if mode == 'avg':
        result = 1 - torch.mean(max_overlap).item()  # Use mean overlap as the result
    elif mode == 'ce':
        result = torch.mean(-torch.log(max_overlap)).item()  # Use cross-entropy-like loss
    elif mode == 'focal':
        result = torch.mean(-(1 - max_overlap) ** 2 * torch.log(max_overlap)).item()  # Focal loss
    else:
        raise ValueError(f'Invalid mode: {mode}')

    if verbose:
        print(f'Current result: {result}')
        print(f'Ratios: {ratios.tolist()}')
        print(f'Scales: {scales.tolist()}')
        print(f'Average overlap: {torch.mean(max_overlap).item()}')
        print('')

    return result


def anchors_optimize(annotations,
                     ratios=3,
                     scales=3,
                     objective='focal',
                     popsize=15,
                     mutation=0.5,
                     image_min_side=800,
                     image_max_side=1333,
                     sizes=[32, 64, 128, 256, 512],
                     strides=[8, 16, 32, 64, 128],
                     include_stride=False,
                     resize=False,
                     threads=1,
                     verbose=True,
                     seed=None):

    # Load annotations and prepare them as in the original code, then perform the optimization

    entries, image_shape = load_annotations(annotations, include_stride)

    bounds = []
    for _ in range(int((ratios - 1) / 2)):
        bounds.append((1, 4))  # Bounds for the ratios
    for _ in range(scales):
        bounds.append((0.4, 2)) 

    result = scipy.optimize.differential_evolution(
        func=average_overlap,
        args=(entries, image_shape, objective, ratios, include_stride, sizes, strides, verbose),
        mutation=mutation,
        updating='deferred' if threads > 1 else 'immediate',
        workers=threads,
        bounds=bounds,
        popsize=popsize,
        seed=seed
    )

    return result


def main():

    path_to_dataset = '/vol/biomedic3/bglocker/mscproj24/mrm123/retinanet/csv_files//vindr_massfil_0ns/train.csv'
    data = pd.read_csv(path_to_dataset)
    data = data[data['numfind'] > 0]

    print("Initializing Anchor Optimization Process")

    optimized_result = anchors_optimize(
        annotations=data,
        ratios=3,   
        scales=3,  
        objective='avg', 
        popsize=15, 
        mutation=0.5, 
        sizes=[32, 64, 128, 256, 512],  
        strides=[8, 16, 32, 64, 128],  
        include_stride=False,  
        resize=False, 
        threads=1,  
        verbose=True,  
        seed=42  
    )

    print("Optimized anchor configuration:", optimized_result)


if __name__=="__main__":
    main()

