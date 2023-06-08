
# Dog Bred Recognition

Proyecto de reconocimiento de diversas razas de perro (133 razas especificas) a traves del uso de redes neuronales y tecnicas de Transfer Learning y Fine Tuning con el uso de un modelo preentrenado como lo es MobileNetV2



## API Reference

#### Obtención de las predicciones

```http
  POST /url/predict
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `File` | `File` | 	**Obligatorio**. Archivo de imagen a procesar. |




## Deployment

Si desea desplegar este proyecto de manera local debera realizar los siguientes pasos para el correcto funcionamiento de la API:

```bash
  pip install -r /path/requirements.txt
```


```bash
  python /path/Fast_dog_api.py
```
**NOTA** TENGA EN CUENTA QUE **PATH** HACE REFERENCIA A LA RUTA DONDE DESCARGUE EL PROYECTO

## Author

- [@ng0-k1](https://github.com/ng0-k1)


## Run Locally

Clone el proyecto

```bash
  git clone https://github.com/ng0-k1/Dog-Breed-Recognition.git
```

Vaya al directorio del proyecto

```bash
  cd my-project
```

Instale los requisitos

```bash
  pip install -r requirements.txt
```

Inicie el servidor
```bash
  python Fast_dog_api.py
```


## Installation

Si desea ejectutar este archivo en un servidor puede recurrir al uso de [Railjack](https://railway.app?referralCode=TcQi7F) y copiar el proyecto y/o modificarlo para resubirlo.

Tenga en cuenta que es de suma importancia **EL USO DEL ARCHIVO modelo_v2_0.h5** ya que este sera el pilar fundamental para las predicciones de las razas


Tambien es importante mencionar que existe una configuración llamada [Procfile](https://github.com/ng0-k1/Dog-Breed-Recognition/blob/main/Procfile) y este archivo contiene la configuración del servidor, se recomienda no realizar modificaciones en este archivo ya que es necesario si usted desea ejecutar este proyecto en [Railjack](https://railway.app?referralCode=TcQi7F)

Aquí se presenta la configuración de este archivo
```bash
  web: uvicorn Fast_dog_api:app --host=0.0.0.0 --port=${PORT:-5000}
```
    

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
