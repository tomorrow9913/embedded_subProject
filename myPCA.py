import numpy as np
import cv2
import copy

# 평균 영상 구하기
def Average_Image(images):
    # resize된 영상의 리스트만을 전달받기 때문에 해당 영상의 가로, 세로 사이즈를 변수에 저장해줌.
    width = images[0].shape[1]
    height = images[0].shape[0]
    # sum
    sum = np.zeros((height, width), dtype=np.float32)
    for i in range(len(images)):
        # 영상을 14개씩 6줄을 출력하고 난 뒤에 모든 영상을 지워줌
        if (i // 14) % 6 == 0 and i % 14 == 0 and i != 0:
            cv2.waitKey(1)
            cv2.destroyAllWindows()

        # 평균 영상을 구하기 위해 sum numpy 배열에 값을 더해줌.
        sum += np.array(images[i], dtype=np.float32)

        # 영상이 정상적으로 열리고 저장이 되고 있는지를 확인하기 위해 모든 영상들을 영상번호에 맞춰서 출력을 반복해줌
        title = f"train{i:03d}.jpg"
        # 이미지들이 겹치지 않도록 위치를 지정해줌
        cv2.namedWindow(title)
        cv2.moveWindow(title, (i % 14) * width, (((i // 14) % 6) * (height + 20)))
        cv2.imshow(title, images[i].astype(np.uint8))

    # 모든 영상을 출력 및 모든 영상의 합을 구한 뒤 출력된 화면들을 모두 제거해줌
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    # 이미지 리스트 파일의 길이를 모든 영상의 합인 sum numpy 배열을 나누어서 평균영상을 구해줌
    return sum / len(images)


# 차영상 구하기
def Difference_Image(images, average):
    # 모든 학습 영상들의 차영상을 구하기 위해 학습 영상 리스트와 평균영상을 전달받음.
    # 평균 영상을 통해 총 화소의 갯수를 계산해줌
    size = average.shape[0] * average.shape[1]
    # 차영상을 담을 numpy 배열을 생성해줌.
    array_image = np.zeros((size, 1), dtype=np.float32)
    for i in range(len(images)):
        # 리스트에서 영상을 꺼내와 해당 영상을 평균 영상으로 빼준 뒤 세로영상으로 변환 해주고 numpy 배열에 추가해줌.
        image = images[i]
        image = image - average
        image = image.reshape(size, 1)
        array_image = np.append(array_image, image, axis=1)

    # numpy 배열 첫 열은 zero값이 들어가 있기 때문에 해당 줄을 지워준 뒤 값을 반환해줌
    return np.delete(array_image, 0, axis=1)


# 고유값/고유벡터 정렬
def Eigen_Sort(value, vector):
    # 고유값을 내림차순으로 정렬한 값을 index 변수에 저장한다.
    index = value.argsort()[::-1]
    # 해당 index를 이용하여 고유값을 정렬하고
    value_sort = value[index]
    # 마찬가지로 index를 이용하여 고유벡터도 정렬한다
    vector_sort = vector[:, index]
    # 정렬 한 고유값과 고유벡터를 반환해준다.
    return value_sort, vector_sort


# 고유값의 주성분 비율을 통해 사용 할 주성분의 갯수를 구하는 함수
def Select_Vector(sort, rate):
    # 정렬 된 고유값에서 실제로 사용할 주성분 만을 가지고 오는 작업
    sum = 0
    # 모든 고유값의 합에 사용할 주성분의 비율인 rate를 곱해주어 기준을 정해줌
    sum_eigen_value = sort.sum() * rate
    for i in range(len(sort)):
        # 고유값의 첫 번째 값 부터 더해가면서 기준을 넘어가거나 같아질 때 해당 Index에 +1을 한 값을 반환해줌
        sum += sort[i]
        if sum_eigen_value <= sum:
            return i + 1


# 변환행렬 축소 구하기
def Transform_Matrix_Reduce(difference_array, eigen_vector_sort, select_index, size):
    # 축소한 변환행렬을 저장해줄 numpy 배열을 선언해줌
    transform_matrix = np.zeros((size, 1))
    # 이전에 구한 주성분의 갯수만큼을 반복해서 돌려줌(0 ~ select_index)
    for i in range(select_index):
        # 차 영상과 고유벡터를 행렬곱해주고, 세로 영상으로 변환해줌
        mul = (difference_array @ eigen_vector_sort[:, i]).reshape(size, 1)
        # 변환된 영상을 정규화 한 뒤 배열에 추가해줌
        transform_matrix = np.append(transform_matrix, mul / np.linalg.norm(mul), axis=1)
    # 처음에 numpy 배열을 생성할 때 만든 0번째 열을 제거한 뒤 반환해줌
    return np.delete(transform_matrix, 0, axis=1)


# 주성분 데이터 투영 구하기
def Calculate_PCA(image_count, difference_array, select_index, transform_matrix):
    # PCA 결과를 저장할 numpy 배열을 선언해줌
    pca_array = np.zeros((select_index, 1))
    for i in range(image_count):
        # 축소한 변환 행렬의 전치를 한 배열과 차이벼열을 행렬곱을 통해여 PCA를 계산하고, PCA numpy배열에 저장합니다.
        pca_array = np.append(pca_array, (transform_matrix.T @ difference_array[:, i]).reshape(select_index, 1), axis=1)

    # 처음에 numpy 배열을 생성할 때 만든 0번째 열을 제거한 뒤에 PCA 결과값을 반환해줌
    return np.delete(pca_array, 0, axis=1)


# PCA 구현 함수
def myPCA(image_files, size, rate):
    image_count = len(image_files)
    # 평균 영상 구하기
    average = Average_Image(copy.deepcopy(image_files))

    cv2.destroyAllWindows()

    cv2.imshow("Average Image", average.astype(np.uint8))

    # 차영상 한줄영상
    print("차 영상 구하기 시작")
    difference_array = Difference_Image(copy.deepcopy(image_files), average)
    print("차 영상 구하기 완료")
    print(difference_array)

    # 공분산 행렬
    print("공분산 행렬 구하기 시작")
    covariance_array = np.cov(difference_array.T)
    print("공분산 행렬 구하기 완료")
    print(covariance_array)

    # 고유값 고유벡터
    print("고유값, 고유벡터 구하기 시작")
    eigen_value, eigen_vector = np.linalg.eig(covariance_array)
    print("고유값, 고유벡터 구하기 완료")
    print(eigen_value)
    print(eigen_vector)

    # 고유값/고유벡터 정렬
    print("고유값/고유벡터 정렬 시작")
    eigen_value_sort, eigen_vector_sort = Eigen_Sort(eigen_value, eigen_vector)
    print("고유값/고유벡터 정렬 완료")
    print(eigen_value_sort)
    print(eigen_vector_sort)

    # 고유벡터 선택
    print("고유벡터 선택 시작")
    select_index = Select_Vector(eigen_value_sort, rate)
    print("고유벡터 선택 완료")
    print(select_index)

    # 변환 행렬 축소
    print("변환 행렬 축소 시작")
    transform_matrix = Transform_Matrix_Reduce(difference_array, eigen_vector_sort, select_index, size)
    print("변환 행렬 축소 종료")
    print(transform_matrix)
    print(transform_matrix.sum())

    # PCA 배열 구하기
    print("주성분 데이터 투영 시작")
    pca_array = Calculate_PCA(image_count, difference_array, select_index, transform_matrix)
    print("주성분 데이터 투영 종료")
    print(pca_array)

    return average, difference_array, select_index, transform_matrix, pca_array


def main():
    num = 10
    count = 40   # 학습 영상의 갯수를 나타내는 변수
    width = 120         # 영상의 가로 사이즈를 결정하는 변수
    height = 150        # 영상의 세로 사이즈를 결정하는 변수
    rate = 0.95         # 주성분의 갯수를 결정할 때 선택 비율을 결정하는 변수
    size = width * height   # 영상의 가로 사이즈와 세로 사이즈를 곱한 값인 18000을 많이 사용하기 위해 선언한 영상 총 사이즈를 나타내는 변수

    # 학습파일을 저장하는 리스트 생성
    files = []

    for i in range(count):

        image = cv2.imread(f"./train_img/{i:03d}.jpg", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (width, height))
        files.append(image)

    for i in range(4):
        average, difference, index, transform, pca_array = myPCA(files[i * num : (i + 1) * num], size, rate)

        cv2.imwrite(f"./train_file/average{i}.jpg", average)
        cv2.imwrite(f"./train_file/difference{i}.jpg", difference)

        f = open(f"./train_file/index{i}.txt", 'w')
        f.write(f"{index},{i * 10}")
        f.close()

        np.save(f"./train_file/transform{i}", transform)
        np.save(f"./train_file/pca_array{i}", pca_array)


if __name__ == '__main__':
    main()
