# Variable
MODEL="efficientdet_lite4"
LITE="lite4"

# Descarga del .tar.gz
curl -L -o ./"${MODEL}".tar.gz  https://www.kaggle.com/api/v1/models/tensorflow/efficientdet/tfLite/"${LITE}"-detection-metadata/1/download

# Crear un directorio temporal para la extraccion
mkdir aux_dir
tar -xzvf "${MODEL}".tar.gz -C aux_dir

# Mover y renombrar el archivo al directorio deseado
mv aux_dir/* ../"${MODEL}".tflite

# Limpiar el directorio temporal
rm -r aux_dir
