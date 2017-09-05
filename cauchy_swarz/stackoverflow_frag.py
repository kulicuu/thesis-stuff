



from PIL import Image
import numpy as npy





img_1 = npy.asarray(Image.open('image_1.jpg').convert('L'))
print(img_1)

img_2 = npy.asmatrix(Image.open('image_1.jpg').convert('L'))
print(img_2, 'img_2')


[x, y] = img_1.shape
print(x, 'x')

N = npy.zeros(shape=(5,5))
print(N, 'N')

# N(3, 3) = 40
N[3][3] = 40
print(N[3][3], 'the entry 3,3')

print(img_2[0:2][0:2], 'img_2 sub')

M = npy.array([[1,2,3], [4,5,6], [7,8,9]])

print(M[0:2:1, 0:2:1])









def naive_corr(pat, img):
    [n, m] = img.shape
    [np, mp] = pat.shape
    N = npy.zeros(shape=(n - np + 1, m - mp + 1))
    for i in range(0, n - np + 1):
        for j in range(0, m - mp + 1):
            img_sub = img[i : i + np : 1, j : j + mp : 1 ]
            N[i][j] = npy.sum(npy.dot(pat, img_sub))
    return N



def box_corr2(img, box_arr, w_arr, n_p, m_p):


    I = (img.cumsum(axis = 0)).cumsum(axis = 1)
    I = npy.array([zeros(1, I.shape[1] + 1)], [zeros()])

    C = npy.zeros(n - n_p, m - m_p)

    jump_x = 1
    jump_y = 1

    x_start = ceil(n_p / 2)
    x_end = n - x_start + npy.mod(n_p, 2)
    x_span = npy.arrange(x_start, x_end, 1)

    y_start = npy.ceil(m_p / 2)
    y_end = m - y_start + npy.mod(m_p, 2)
    y_span = npy.arrange(y_start, y_end, 1)

    arr_a = box_arr







def naive_normxcorr2(temp, img):
    return 0