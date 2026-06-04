# Generacion de contenido - TFM – IA Forense

---

A continuación, se mostrará algunos ejemplos de contenido generado mediante inteligencia artificial, con el objetivo de ilustrar de manera práctica el alcance y realismo que estas tecnologías pueden alcanzar en la actualidad. Este análisis se realizará bajo un enfoque estrictamente académico y controlado, respetando consideraciones éticas y legales, y sin incurrir en la creación o difusión de contenido que pueda vulnerar derechos o generar desinformación. Como aclaración, se evidencia que para obtener los mejores resultados era necesario pagar una suscripción premium o ejecutar de modo local los modelos generativos (para lo cual es necesario un computador de alta gama). Esto se aclara porque los mejores resultados vienen ligados de esas dos condiciones, por lo tanto, no lograre mostrar los resultados más realistas y se citaran personas que SI hayan podido tener alguna de las condiciones previamente descritas.

---

## 1. Imágenes generadas por IA

La generación de imágenes mediante inteligencia artificial ha alcanzado un nivel de realismo que, en muchos casos, resulta indistinguible para un observador no entrenado. A través de modelos generativos avanzados, es posible crear escenas, rostros y contextos completamente ficticios a partir de simples instrucciones en lenguaje natural. Esta sección tiene como objetivo evidenciar, mediante ejemplos prácticos, cómo estas herramientas pueden producir contenido visual creíble.

### 1.1 Sección 1: “One Click” (Modelos tipo ChatGPT y Gemini):

En primer lugar, se evaluaron herramientas de generación de imágenes accesibles mediante interfaces conversacionales, comúnmente denominadas como soluciones “one click”, como las ofrecidas por ChatGPT y Google Gemini. Estas plataformas integran modelos generativos de imágenes que permiten, a partir de un prompt, obtener resultados en cuestión de segundos sin necesidad de configuraciones avanzadas. En el caso de ChatGPT, se apoya en modelos como DALL·E y recientemente en GPT-4º [19], mientras que Gemini emplea arquitecturas propias (por ejemplo, variantes internas como Nano Banana 2 [20]).

Los resultados obtenidos fueron mixtos. En algunos casos, los modelos lograron generar imágenes realistas o similares, especialmente cuando los prompts incluían condiciones detalladas como iluminación, imperfecciones o contexto (por ejemplo, estilo CCTV, fotografía casual o evidencia tipo mugshot). Sin embargo, en otros escenarios, se evidenciaron limitaciones claras, como la tendencia a replicar la pose original de la imagen base o mantener fondos similares, lo que puede delatar el uso de inteligencia artificial. Para esta evaluación, se utilizaron múltiples prompts diseñados específicamente para forzar cambios de ángulo, contexto y condiciones, los cuales se presentan como ejemplos prácticos del comportamiento de estos modelos. A continuacion el detalle:

**ChatGPT – Escenario cafetería**
![ChatGPT Cafetería](ChatGPTCafeteria.png)

**ChatGPT – Escenario CCTV**
![ChatGPT CCTV](ChatGPTCCTV.png)

**ChatGPT – Escenario incidente**
![ChatGPT Incidente](ChatgptIncidente.png)

**ChatGPT – Selfie**
![ChatGPT Selfie](ChatGPTSelfie.png)

**ChatGPT – Estilo años 80**
![ChatGPT Los80](ChatgptLos80.png)

**ChatGPT – Entorno oficina**
![ChatGPT Oficina](ChatgptOficina.png)

**ChatGPT – Documento tipo pasaporte**
![ChatGPT Pasaporte](Chatpasaporte.png)

---

**Gemini – Escenario cafetería**
![Gemini Cafetería](geminicafeteria.png)

**Gemini – Escenario CCTV**
![Gemini CCTV](geminicctv.png)

**Gemini – Fotografía envejecida**
![Gemini Foto vieja](geminifotovieja.png)

**Gemini – Documento tipo pasaporte**
![Gemini Pasaporte](geminipasaporte.png)

**Gemini – Escenario policial**
![Gemini Policía](geminipolicia.png)

**Gemini – Selfie**
![Gemini Selfie](geminiselfie.png)

**Gemini – Entorno laboral**
![Gemini Trabajo](geminitrabajo.png)

---

### 1.2 Sección 2: Modelos personalizados (fine-tuning / entrenamiento)

En un segundo nivel de complejidad se encuentran los modelos personalizados, los cuales permiten entrenar una inteligencia artificial sobre un individuo específico. Un modelo corresponde a una red neuronal previamente entrenada que puede ajustarse mediante técnicas de fine-tuning [21], utilizando imágenes adicionales para especializar la generación de contenido de una persona concreta. Actualmente, este proceso resulta altamente accesible gracias a plataformas que automatizan gran parte del entrenamiento. Como parte de esta investigación, se desarrolló un modelo propio utilizando Higgsfield 2.0, entrenado con aproximadamente 60 imágenes en distintos ángulos y contextos. El resultado fue un modelo capaz de generar imágenes realistas en múltiples escenarios, logrando engañar incluso a familiares y personas cercanas. 

Esta parte se lleva a cabo únicamente con fotos en Higgsfield, con cada iteración mejora.

**Iteración inicial 1**
![PreModelo 1](PreModelo1.png)

**Iteración inicial 2**
![PreModelo 2](Premodelo2.png)

**Iteración inicial 3**
![PreModelo 3](Premodelo3.png)

**Iteración inicial 4**
![PreModelo 4](Premodelo4.png)

**Iteración inicial 5**
![PreModelo 5](Premodelo5.png)

**Iteración inicial 6**
![PreModelo 6](Premodelo6.png)

---

### 1.3 Modelo

Con cada iteración el modelo mejoraba, dando resultados más fieles a la realidad. Dichas imágenes fueron sometidas a una prueba, donde se las envié a mis seres queridos, y adjudicaron que eran ciertas (a excepción del contexto irreal del presaje). 

**Escenario carcelario**
![Modelo Carcel](ModeloCarcel.png)

**Escenario deportivo**
![Modelo Futbol](ModeloFutbol.png)

**Perfil joven**
![Modelo Joven](ModeloJoven.png)

**Escenario penitenciario 1**
![Modelo Preso 2](Modelopreso2.png)

**Escenario penitenciario 2**
![Modelo Preso 3](Modelopreso3.png)

**Selfie generado 1**
![Modelo Selfie 1](ModeloSelfie1.png)

**Selfie generado 2**
![Modelo Selfie 2](ModeloSelfie2.png)

**Escenario de viaje 1**
![Modelo Viaje](ModeloViaje.png)

**Escenario de viaje 2**
![Modelo Viaje 2](ModeloViaje2.png)

La IA puede “modelar” mi rostro y reusarlo en distintos planos, aunque este modelaje aun falla un poco rostro frontal, pero no en el perfil (al menos en las fotos de preso que se pueden ver en el repositorio). Finalmente, en diversas plataformas en línea, como ComfyUI o Civitai, es posible acceder a modelos previamente entrenados y listos para su uso, tanto de personas reales como de representaciones artificiales de personajes ficticios. Estos modelos, disponibles de forma abierta o mediante servicios específicos, permiten generar contenido sin necesidad de realizar procesos de entrenamiento desde cero, por ejemplo: 


---

## 2. Videos

### 2.1 FaceSwap 

Se presentan ejemplos de intercambio facial (FaceSwap), comparando contenido original y manipulado.

## Hombre

**FaceSwap – Video original**
[![FaceSwap Hombre Original](CapturaFaceSwapOriginal.png)](https://drive.google.com/file/d/1ZXYr6qGf3ulQakvt4q8N7sSzZBsKTlme/view?usp=sharing)

**FaceSwap – Video manipulado**
[![FaceSwap Hombre Falso](CapturaFaceSwapFake.png)](https://drive.google.com/file/d/1rBTiMZwzqjD8EFrGir4WMaPVuWYu0mXl/view?usp=sharing)

## Mujer

**FaceSwap – Video original**
[![FaceSwap Mujer Original](CapturaFaceSwapMujerOriginal.png)](https://drive.google.com/file/d/1o0kBUXyWYKkR_dkoQeCnClc5htozcjNG/view?usp=sharing)

**FaceSwap – Video manipulado**
[![FaceSwap Mujer Falso](CapturaFaceSwapMujerFake.png)](https://drive.google.com/file/d/16_q8zwRuMK0H92OwmunFzDe8XYRKobDC/view?usp=sharing)

---

### 2.2 Deepfakes

Ejemplos de contenido generado mediante técnicas de deepfake, comparando versiones originales y alteradas.

**Deepfake – Contenido generado**
[![Deepfake Original](CapturaDeepfakeFalso.png)](https://drive.google.com/file/d/1IvJi4stSvZm4z5y5XWb1lZCv1va5lVx7/view?usp=drive_link)

**Deepfake – Contenido original**
[![Deepfake Original](CapturaDeepfakeOriginal.png)](https://drive.google.com/file/d/191q9wbfQwqFfB4K1YNtk-1_EU32aIqAu/view?usp=sharing)

---

### 2.3 Videos de modelo

Videos generados a partir del modelo entrenado en distintos escenarios.

**Escenario caminando**
[![Modelo Caminar](Yo.jpeg)](https://drive.google.com/file/d/1wv3prD-Btp7SsVxJi2tfW30gEMdSwc0Z/view?usp=sharing)

**Escenario entrada**
[![Modelo Entrar](Yo.jpeg)](https://drive.google.com/file/d/1KkZXczo5qYlVuyqWmiIEW_W_u9m4kFje/view?usp=sharing)

**Escenario nocturno**
[![Modelo Noche](Yo.jpeg)](https://drive.google.com/file/d/1l8IRU6pZ4udmIkmUDsBhWgJDwgfGL_cN/view?usp=sharing)

**Escenario tipo noticia**
[![Modelo Noticia](Yo.jpeg)](https://drive.google.com/file/d/1suxETh_D6xMRw0SutS-YxKAgtLgl-hQ5/view?usp=sharing)

**Escenario robo**
[![Robo](Yo.jpeg)](https://drive.google.com/file/d/1rikHtr5qWcknicHH4tm6CKqzEi4-gUgd/view?usp=sharing)

---

### 2.4 Character Motion

Ejemplos de animación de personajes mediante técnicas de motion transfer.

**Animación generada**
[![Character Motion Fake](CapturaCharacterFake.png)](https://drive.google.com/file/d/19FtIG-wEBo7ykzlOGZ3-DtrQ5bgj7sTX/view?usp=sharing)

**Animación original**
[![Character Motion Original](CapturaCharacterOriginal.png)](https://drive.google.com/file/d/11zVWjsy7Kwbuft0IYOlD11uUGp8c4Ppa/view?usp=sharing)

---

### 2.5 Edición de video

Ejemplos de manipulación y edición de contenido audiovisual.

**Edición – Ejemplo**
[![Edición Video Ejemplo](CapturaEdicionEj.png)](https://drive.google.com/file/d/1v741IYfdKkE2Tg9A6jbeKey-M7RKdTXI/view?usp=sharing)

**Edición – Resultado final**
[![Video Edición](CapturaEdicionVideo.png)](https://drive.google.com/file/d/1-nWQvaD3aifi7A2ev4Vcry1TSuOzFK6e/view?usp=sharing)

---

## 3. Audios

### 3.1 Audio original

Registro de audio sin modificación.

[Escuchar audio](https://drive.google.com/file/d/18b_-tmXc_Abuz-rtacHwuCOtrLE42plB/view?usp=sharing)

---

### 3.2 Audio modelo

Audio generado o modificado mediante modelo de IA.

[Escuchar audio](https://drive.google.com/file/d/1pqG08NTEPX1Kz3NEoGWAG_TYhzY7OAXa/view?usp=sharing)

---

### 3.3 Audio FakeYou

Ejemplo de clonación de voz mediante plataforma FakeYou.

[Escuchar audio](https://drive.google.com/file/d/1DxMua5JDreN7BNpGaWUaudmkq3TUCl-r/view?usp=sharing)

---

### 3.4 Canciones generadas

Contenido musical generado mediante modelos de IA.

[Escuchar canción](https://drive.google.com/file/d/1arji4z_0NK7xuwhHvBsbdtiao1NDVa6a/view?usp=sharing)

[Escuchar canción](https://drive.google.com/file/d/1oWNGchH9uAu90rDjJSl0CjdoGxQvDmuC/view?usp=sharing)

---

## 4. Edición de imágenes con IA

Ejemplos de modificaciones realizadas sobre imágenes mediante herramientas de IA.

![Edición 1](Edicion1.png)
![Edición 2](Edicion2.png)
![Edición 3](Edicion3.png)
![Edición 4](Edicion4.png)

---

## 5. Avatares generados con IA

Ejemplos de generación de avatares mediante modelos de inteligencia artificial.

[![Avatar IA](CapturaAvatarIA.png)](https://drive.google.com/file/d/1q4Cit47Syr5k8JGOGdokKsqSf37htQfS/view?usp=sharing)

[![Avatar Deepfake](CapturaAvatarDeepfake.png)](https://drive.google.com/file/d/1I90-R4NeN29ZUiD5mmUvtpGWTWuG-QbG/view?usp=sharing)
