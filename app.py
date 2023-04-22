import cv2
from flask import Flask, render_template, Response

# Cargamos el clasificador de rostros pre-entrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializamos la aplicación Flask
app = Flask(__name__)

# Definimos una ruta raíz para mostrar la página HTML
@app.route('/')
def index():
    return render_template('index.html')

# Definimos una función que se ejecutará en segundo plano para procesar los datos de la cámara
def generate_frames():
    # Abrimos la cámara
    cap = cv2.VideoCapture(0)

    while True:
        # Leemos el frame actual
        ret, frame = cap.read()

        if not ret:
            break

        # Convertimos a escala de grises para la detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectamos los rostros en la imagen
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Dibujamos un rectángulo alrededor de cada rostro detectado
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Codificamos el frame en formato JPEG para transmitirlo a través de la red
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Generamos un objeto de respuesta de Flask con el frame codificado
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Liberamos los recursos utilizados por la cámara
    cap.release()

# Definimos una ruta para transmitir los frames a través de la red
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Iniciamos la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)