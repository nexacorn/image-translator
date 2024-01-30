import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN, KMeans
from app.module.utils import calculate_center, color_similarity, adjust_bbox


def cluster_boxes_by_text_height(boxes, eps):
    # Extract text heights
    text_heights = np.array([box[3] for box in boxes]).reshape(-1, 1)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=1, n_jobs=-1).fit(text_heights)
    labels = clustering.labels_

    # Group boxes by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label in clusters:
            clusters[label].append(i)
        else:
            clusters[label] = [i]

    for cluster in list(clusters.values()):
        cluster_text_heights = [boxes[i][3] for i in cluster]
        min_height = min(cluster_text_heights)

        for i in cluster:
            boxes[i][3] = min_height  # Update the height of the box to the mean height
    return boxes


def cluster_boxes_by_text_y(boxes, eps):
    # Extract text y coordinates
    text_y_coords = np.array([box[1] for box in boxes]).reshape(-1, 1)

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=1, n_jobs=-1).fit(text_y_coords)
    labels = clustering.labels_

    # Group boxes by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label in clusters:
            clusters[label].append(i)
        else:
            clusters[label] = [i]

    for cluster in list(clusters.values()):
        cluster_centers = [calculate_center(boxes[i]) for i in cluster]
        mean_center = np.mean(cluster_centers, axis=0)

        mean_y = int(mean_center[1]) - int(boxes[cluster[0]][3] / 2)

        for i in cluster:
            boxes[i][1] = mean_y

    return boxes


def cluster_colors(data, eps):
    # 입력 데이터에서 font_color를 추출
    colors = [item[6] for item in data]

    # RGB 색상 간의 유사성을 기반으로 DBSCAN 클러스터링 수행
    clustering = DBSCAN(eps=eps, min_samples=1, metric=color_similarity, n_jobs=-1).fit(
        colors
    )
    labels = clustering.labels_

    # 각 클러스터에서 가장 진한 색상 선택
    clusters = {}
    for i, label in enumerate(labels):
        if label in clusters:
            clusters[label].append(i)
        else:
            clusters[label] = [i]

    dominant_colors = []
    for cluster in clusters.values():
        cluster_colors = [colors[i] for i in cluster]
        # 가장 진한 색상 선택 (RGB 값의 합이 가장 큰 것)
        color_mean = np.mean(cluster_colors)
        if color_mean <= 125.0:
            dominant_color = min(cluster_colors, key=lambda color: sum(color))
        else:
            dominant_color = max(cluster_colors, key=lambda color: sum(color))
        dominant_colors.append(dominant_color)

    # 입력 데이터의 font_color를 클러스터링 결과로 대체
    for i, item in enumerate(data):
        item[6] = dominant_colors[labels[i]]

    return data


def align_text_boxes(boxes, input_image, center_eps=80, x_eps=20):
    image = input_image

    # 1단계: 센터 좌표 계산
    centers = np.array([(x + w / 2, y + h / 2) for x, y, w, h, text, sim, _ in boxes])

    # 2단계: DBSCAN을 사용하여 센터 기반 클러스터링
    clustering = DBSCAN(eps=center_eps, min_samples=1, n_jobs=-1).fit(centers)
    labels = clustering.labels_

    # 3단계: 각 섹션(클러스터) 내에서 x 좌표에 대한 클러스터링
    new_boxes = []
    for label in set(labels):
        cluster_boxes = [box for box, l in zip(boxes, labels) if l == label]
        cluster_centers = [center for center, l in zip(centers, labels) if l == label]

        # 섹션 내의 x 좌표에 대한 클러스터링
        x_coords = np.array([center[0] for center in cluster_centers])
        x_clustering = DBSCAN(eps=x_eps, min_samples=1, n_jobs=-1).fit(
            x_coords.reshape(-1, 1)
        )
        x_labels = x_clustering.labels_

        # 각 x 클러스터에 대한 평균 x 좌표 계산 및 박스 x 좌표 재설정
        for x_label in set(x_labels):
            x_cluster_boxes = [
                box for box, x_l in zip(cluster_boxes, x_labels) if x_l == x_label
            ]

            avg_center_x = np.mean(
                [
                    center[0]
                    for center, x_l in zip(cluster_centers, x_labels)
                    if x_l == x_label
                ]
            )

            # 갱신된 x 좌표로 boxes 재구성
            new_boxes.extend(
                [
                    (int(avg_center_x - w / 2), y, w, h, text, sim, _)
                    for x, y, w, h, text, sim, _ in x_cluster_boxes
                ]
            )
    return new_boxes


def extract_border_colors(input_image, bbox, expansion=1, n_clusters=1):
    # 이미지 로드
    image = input_image
    image = np.array(image)

    x1, y1, x2, y2 = adjust_bbox(image.shape, bbox)
    x1, y1, x2, y2 = (
        max(x1 - expansion, 0),
        max(y1 - expansion, 0),
        min(x2 + expansion, image.shape[1]),
        min(y2 + expansion, image.shape[0]),
    )

    # 경계선 색상 추출을 위한 인덱스 조정
    if x2 >= image.shape[1]:
        x2 = image.shape[1] - 1

    if y2 >= image.shape[0]:
        y2 = image.shape[0] - 1

    # 경계선 색상 추출
    top_border = image[y1, x1 : x2 + 1]
    bottom_border = image[y2, x1 : x2 + 1]
    left_border = image[y1 : y2 + 1, x1]
    right_border = image[y1 : y2 + 1, x2]

    border_colors = np.vstack((top_border, bottom_border, left_border, right_border))

    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(border_colors)
    return kmeans.cluster_centers_


def extract_bbox_colors(input_image, bbox, n_clusters=2):
    # 이미지 로드
    image = input_image
    image = np.array(image)

    x1, y1, x2, y2 = adjust_bbox(image.shape, bbox)

    # 바운딩 박스 내부 픽셀 추출
    bbox_pixels = image[y1 : y2 + 1, x1 : x2 + 1].reshape(-1, 3)

    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(bbox_pixels)
    return kmeans.cluster_centers_
