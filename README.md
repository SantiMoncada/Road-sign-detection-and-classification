# Trabajo de fin de grado de Santiago Moncada

# DESARROLLO DE SISTEMA PARA ASISTIR A LA CONDUCCIÓN AUTÓNOMA MEDIANTE CÁMARAS

## Preparación del entorno

### Instalación Python y librerías

Es necesario instalar Python 3 para el funcionamiento del programa, se puede conseguir en la siguiente dirección: https://www.python.org/downloads/

Las librerías que necesitaremos para ejecutar el proyecto son:

Numpy, OpenCV y Tensor Flow, la cual instalaremos con los comandos:
```
pip3 install numpy

pip3 install opencv-python

pip3 install tensorflow
```
Una vez que tenemos todas las dependencias solucionadas, podemos ejecutar los scripts de clasificación con los comandos de:
```

python camaraTiempoReal.py

Python ProcesarVideo.py

Python notGui.py
```

Si lo que deseamos es entrenar el sistema con los datos, proporcionamos, tenemos más dependencias.

Para entrenar el modelo de clasificación CNN necesitamos las librerías Pandas, Matplotlib, PIL, Sklearn:

```
pip3 install pandas

pip3 install Matplotlib

pip3 install pillow

pip install scikit-learn
```
 

Para entrenar el modelo de identificación YOLO, debemos utilizar la implementación de darknet y para facilitar mucho las cosas es recomendable ejecutarla en linux ubuntu:
```
git clone https://github.com/AlexeyAB/darknet.git
```
Entramos en el directorio del programa:
```
cd darknet
```
Construimos el programa:
```
make
```
Si todo funciona al ejecutar la línea de código:
```
./darknet
```
Debería generar como resultado: “usage: ./darknet ”

Y con esto ya podemos entrenar el sistemas con los archivos proporcionados con el comando:
```
./darknet detector train <path archivo .data> <path archivo .cfg> <path de weights>
```
Como no tenemos ninguna weights lo utilizara para guardar backups de las iteraciones.