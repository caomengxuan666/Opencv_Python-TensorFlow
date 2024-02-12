from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import cv2
import numpy as np


def encrypt_image(input_image_path, output_image_path, key):
    # Read the image
    image = cv2.imread(input_image_path)

    # Convert the image to bytes
    image_bytes = cv2.imencode(".png", image)[1].tobytes()

    # Create AES encryption object
    cipher = AES.new(key, AES.MODE_CBC)

    # Encrypt the image bytes
    ciphertext = cipher.iv + cipher.encrypt(pad(image_bytes, AES.block_size))

    # Save the encrypted data to a file
    with open(output_image_path, 'wb') as f:
        f.write(ciphertext)

    print("Image encrypted successfully")


def decrypt_image(encrypted_image_path, output_image_path, key):
    # Read the encrypted data from the file
    with open(encrypted_image_path, 'rb') as f:
        ciphertext = f.read()

    # Extract IV from the encrypted data
    iv = ciphertext[:16]

    # Create AES decryption object
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Decrypt the data
    decrypted_bytes = unpad(cipher.decrypt(ciphertext[16:]), AES.block_size)

    # Convert the decrypted bytes to an image
    decrypted_image = cv2.imdecode(np.frombuffer(decrypted_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Save the decrypted image
    cv2.imwrite(output_image_path, decrypted_image)

    print("Image decrypted successfully")


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 生成16字节的随机密钥
key = get_random_bytes(16)

# 输入图像的地址
image_path = 'C:/Users/Lenovo/Desktop/pictures/slfn.jpg'

# 加密保存
encrypt_image(image_path, 'C:/Users/Lenovo/Desktop/pictures/encrypted_image_python.png', key)

# 解密保存
decrypt_image('C:/Users/Lenovo/Desktop/pictures/encrypted_image_python.png',
              'C:/Users/Lenovo/Desktop/pictures/decrypted_image_python.png', key)

dec = cv2.imread('C:/Users/Lenovo/Desktop/pictures/encrypted_image_python.png')

out = cv2.imread('C:/Users/Lenovo/Desktop/pictures/decrypted_image_python.png')


img=cv2.imread('C:/Users/Lenovo/Desktop/pictures/slfn.jpg')

cv_show('img',img)
cv_show('out',out)


print("当前密钥:", key)
