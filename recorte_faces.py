import cv2
import os
from tqdm import tqdm

# Locas das fotos para serem processadas
path_imgs_processar = "processar/"
path_imgs_processadas = "processadas/"

# Raio do rosto
raio_rosto = 150

# Lista as Imagens para serem processadas
imgs = os.listdir(path_imgs_processar)

# Contador de imagens processadas
contador_img = 1

# Tqdm
t = tqdm(imgs)

# Para cada imagem na origem
for img in t:

    # Arquivo que está sendo processado
    t.set_description(f'Arquivo: {img}')

    # Carrega a imagem
    imagem = cv2.imread(path_imgs_processar + img)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplica o filtro
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        cinza,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(100, 100)
    )

    # Contador de faces encontradas
    cont_faces_detectadas = 0

    # Para cada face encontrada
    for (x, y, w, h) in faces:

        # Incrementa as faces detectadas
        cont_faces_detectadas += 1

        # Centraliza a face
        posX = x + (w/2)
        posY = y + (h/2)

        # Recorta a face e salva na pasta de destino
        face = imagem[int(posY - raio_rosto):int(posY + raio_rosto), int(posX - raio_rosto):int(posX + raio_rosto)]
        cv2.imwrite(path_imgs_processadas + str(cont_faces_detectadas) + img, face)


    # Contador de imagens processadas
    contador_img += 1