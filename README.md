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

**Modelo Gratuita**

![Chica](Chica.png)

---
### 1.4 Sección 3: Casos reales y evidencia mediática

Finalmente, se presenta una recopilación de casos reales en los que imágenes generadas mediante inteligencia artificial lograron engañar a audiencias a nivel global. Estos ejemplos evidencian el nivel de realismo alcanzado por los modelos actuales y cómo pueden influir en la desinformación y manipulación de narrativas. Se invita al lector a revisar dichos casos y evaluar por sí mismo la dificultad de distinguir entre contenido real y sintético.

•	https://www.teleantioquia.co/entretenimiento/6-imagenes-hechas-con-inteligencia-artificial-que-confundieron-al-mundo-y-algunas-llegaron-a-generar-escandalos-globales-518763
•	https://www.xataka.com/robotica-e-ia/eres-capaz-distinguir-imagen-real-generada-ia-aqui-tienes-20-fotos-para-comprobarlo
---

## 2. Videos

La generación y manipulación de video mediante inteligencia artificial representa uno de los avances más complejos y, a su vez, más críticos desde una perspectiva forense. A diferencia de las imágenes estáticas, el video incorpora múltiples dimensiones como el movimiento, la coherencia temporal y la sincronización audiovisual, lo que incrementa tanto su realismo como su capacidad de engaño. 

### 2.1 Sección 1: Face Swap (herramientas directas) 

En primer lugar, se evaluaron herramientas de face swap, tanto locales como en línea, las cuales permiten reemplazar el rostro de una persona en imágenes o videos mediante modelos de reconocimiento facial y redes neuronales entrenadas para mapear características faciales sobre una estructura base. Los resultados obtenidos fueron mixtos y dependieron principalmente de la calidad de la herramienta y del nivel de suscripción utilizado. Como ejemplo, se presenta un face swap realizado entre un video del deportista Cristiano Ronaldo y mi persona.

## Hombre

**FaceSwap – Video original**
[![FaceSwap Hombre Original](CapturaFaceSwapOriginal.png)](https://drive.google.com/file/d/1ZXYr6qGf3ulQakvt4q8N7sSzZBsKTlme/view?usp=sharing)

**FaceSwap – Video manipulado**
[![FaceSwap Hombre Falso](CapturaFaceSwapFake.png)](https://drive.google.com/file/d/1rBTiMZwzqjD8EFrGir4WMaPVuWYu0mXl/view?usp=sharing)

## Mujer

En versiones gratuitas, se observaron limitaciones evidentes en la alineación facial, inconsistencias en iluminación y pérdida de naturalidad en el movimiento. Si bien es posible obtener resultados aceptables, alcanzar un nivel alto de realismo requiere acceso a versiones premium o configuraciones más avanzadas. Por esta razón, se presenta un ejemplo del máximo resultado alcanzado bajo estas condiciones,  consiste en la actriz Liv Tyler, haciendo FaceSwap en el video objetivo, con la foto mostrada a continuación como referencia.

![Liv](Liv.jpg)

**FaceSwap – Video original**
[![FaceSwap Mujer Original](CapturaFaceSwapMujerOriginal.png)](https://drive.google.com/file/d/1o0kBUXyWYKkR_dkoQeCnClc5htozcjNG/view?usp=sharing)

**FaceSwap – Video manipulado**
[![FaceSwap Mujer Falso](CapturaFaceSwapMujerFake.png)](https://drive.google.com/file/d/16_q8zwRuMK0H92OwmunFzDe8XYRKobDC/view?usp=sharing)

---

### 2.2 Sección 2: Deepfakes

En un segundo nivel, se analizaron los denominados deepfakes, los cuales consisten en superponer un rostro y, en algunos casos, una voz sobre un video base. Estos sistemas utilizan modelos de Deep Learning, como autoencoders, GANs o arquitecturas híbridas, capaces de transferir expresiones, iluminación y movimiento entre individuos.
La generación de deepfakes de alta calidad requiere mayores recursos computacionales y modelos previamente entrenados. En escenarios avanzados, es posible realizar deepfakes en tiempo real, aplicando un rostro generado sobre un flujo de video en vivo y modificando simultáneamente la voz mediante técnicas voice-to-voice. A continuación, se presenta un ejemplo donde, sobre mi cámara en tiempo real, se superpone el rostro del deportista Cristiano Ronaldo.

**Deepfake – Contenido generado**
[![Deepfake Original](CapturaDeepfakeFalso.png)](https://drive.google.com/file/d/1IvJi4stSvZm4z5y5XWb1lZCv1va5lVx7/view?usp=drive_link)

**Deepfake – Contenido original**
[![Deepfake Original](CapturaDeepfakeOriginal.png)](https://drive.google.com/file/d/191q9wbfQwqFfB4K1YNtk-1_EU32aIqAu/view?usp=sharing)

El nivel de sofisticación suele estar limitado a cuentas premium o a implementaciones locales con alta capacidad computacional (GPU de alto rendimiento), lo que marca una diferencia significativa frente a soluciones básicas. Por lo tanto, adjunto ejemplos de un deepfake llevado a su máximo potencial.
•	https://www.instagram.com/reel/DUDpBmNDUZ_/
•	https://www.youtube.com/watch?v=EXmMPHwSMxI
•	https://www.youtube.com/watch?v=TltS5ZbGtTA
•	https://www.facebook.com/joseantonioponton/videos/cada-vez-son-más-fáciles-de-hacer-los-deepfakes-kling-ia-deepfake-aimodels-tecno/2060236378132252/
Por otro lado, adjunto un recopilado de noticias de intereses sobre deepfakes que fueron noticia por ser engaño, las cuales no se pueden anexar al presente informe debido a que no se divulgo el contenido original del deepfake. https://hyperverge.co/blog/examples-of-deepfakes/

---

### 2.3 Sección 3: Generación de video desde cero (modelos personalizados)

En esta sección se exploró la generación de video completamente sintético utilizando el modelo previamente entrenado en Higgsfield 2.0. A diferencia del face swap o los deepfakes, el contenido no parte de un video base, sino que es generado desde cero mediante instrucciones (prompts) y un modelo personalizado del sujeto.
Los resultados mejoraron progresivamente conforme se optimizaron los prompts. Aunque aún se identifican limitaciones en movimientos y detalles finos, el nivel de realismo alcanzado resulta suficiente para generar contenido creíble, especialmente en contextos no controlados. Además, las versiones premium de estas herramientas prometen mejoras significativas en estabilidad y calidad visual. Todos los videos ejemplo parten de esta foto como guía, y el modelo que se entrenó:

![Yo](Yo.jpeg)

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

Los resultados obtenidos en la generación de video dependen en gran medida de la calidad del prompt utilizado. Aun así, los modelos logran captar adecuadamente rasgos faciales, perfil y atuendo del sujeto. Las versiones premium permiten generar tomas más naturales, mejores movimientos de cámara y condiciones de iluminación más realistas. Asimismo, el uso de hardware potente influye directamente en la estabilidad y calidad visual del resultado.

Para ilustrar el nivel actual de estas tecnologías, resulta pertinente mencionar un caso que generó gran impacto mediático. En el contexto del conflicto entre Irán e Israel, surgieron rumores sobre la supuesta muerte del líder israelí Benjamin Netanyahu [24]. Posteriormente, se difundieron videos destinados a demostrar que seguía con vida; sin embargo, diversos analistas cuestionaron su autenticidad y plantearon la posible intervención de inteligencia artificial en su creación. A continuación, se incluye uno de los análisis realizados sobre este caso: https://www.instagram.com/reel/DV_sYkMDKss/, así como una fuente periodística que documenta el caso: https://www.nytimes.com/2026/03/17/technology/netanyahu-ai-video-iran-israel.html.
De igual forma, otro ejemplo relevante es un cortometraje generado mediante IA que recrea la apariencia, voz y estilo de múltiples creadores de contenido, siguiendo una estructura similar a los conocidos “Rewinds”. El material ha generado repercusión por su capacidad para sincronizar voz, imagen y narrativa de forma convincente, aunque aún presenta detalles que evidencian su naturaleza sintética. El video puede consultarse en: https://www.youtube.com/watch?v=JWiSiDX4oBE.

Por otro lado, herramientas como Sora representaron un avance significativo en la generación de video, destacándose por reproducir movimientos, iluminación e imperfecciones naturales propias de escenas reales. Aunque a la fecha de redacción su acceso ha sido limitado [25], los ejemplos disponibles permiten dimensionar el nivel de realismo alcanzado por estas tecnologías: https://hipertextual.com/tecnologia/14-videos-hechos-con-sora-2-chatgpt-que-te-van-a-volar-la-cabeza/.

---

### 2.4 Sección 4: Motion Control (animación a partir de imagen)

Otra técnica evaluada fue el motion control, que consiste en mapear una imagen sobre un video base, transfiriendo el movimiento corporal y gestual mediante técnicas de pose estimation [22].
Los resultados dependieron principalmente de la compatibilidad entre la imagen y el video utilizado. Factores como orientación facial y estructura corporal influyen directamente en la calidad final. Cuando existe coherencia entre ambos elementos, el resultado puede ser altamente realista; de lo contrario, suelen aparecer artefactos visibles que evidencian la manipulación.

**Animación generada**

[![Character Motion Fake](CapturaCharacterFake.png)](https://drive.google.com/file/d/19FtIG-wEBo7ykzlOGZ3-DtrQ5bgj7sTX/view?usp=sharing)

**Animación original**

[![Character Motion Original](CapturaCharacterOriginal.png)](https://drive.google.com/file/d/11zVWjsy7Kwbuft0IYOlD11uUGp8c4Ppa/view?usp=sharing)

A continuación, se presentan ejemplos de motion control ejecutados mediante herramientas en su versión premium, los cuales permiten apreciar el nivel de calidad que puede alcanzarse bajo condiciones óptimas de ejecución:
•	https://www.youtube.com/watch?v=h3nEVI8rXxk
•	https://www.youtube.com/watch?v=jXBKljs35Yc
•	https://www.youtube.com/watch?v=imn4M5Bws7k
•	https://www.tiktok.com/@elmundoderabbit/video/7596743366147493128
Estos materiales evidencian cómo es posible obtener resultados realistas, especialmente cuando se cumplen las condiciones de compatibilidad entre la imagen y el video base.

---

### 2.5 Sección 5: Edición de video mediante IA

Finalmente, se contempla la edición de video mediante inteligencia artificial, una capacidad que permite modificar aspectos como iluminación, entorno, condiciones climáticas o incluso elementos dentro de la escena. Este tipo de herramientas representa un nivel avanzado de manipulación, ya que no solo genera contenido, sino que altera material existente de forma coherente. A continuación, se presenta un ejemplo donde el video anterior (el mío caminando de día) se edita con una camiseta roja, así como un ejemplo ilustrativo. 

**Edición – Ejemplo**
[![Edición Video Ejemplo](CapturaEdicionEj.png)](https://drive.google.com/file/d/1v741IYfdKkE2Tg9A6jbeKey-M7RKdTXI/view?usp=sharing)

**Edición – Resultado final**
[![Video Edición](CapturaEdicionVideo.png)](https://drive.google.com/file/d/1-nWQvaD3aifi7A2ev4Vcry1TSuOzFK6e/view?usp=sharing)

No obstante, en el desarrollo de esta investigación no fue posible realizar pruebas prácticas debido a que la mayoría de estas funcionalidades se encuentran restringidas a planes premium (y las gratuitas poseen muchos fallos, como lo adjunto en el video). Por esta razón se incluyen ejemplos ilustrativos que permite evidenciar el potencial de estas tecnologías y su impacto en la manipulación de contenido audiovisual: 
•	https://www.youtube.com/watch?v=I_j7MDstows
•	https://www.youtube.com/watch?v=tDt4VZrod3w

---

## 3. Audios

La generación y manipulación de audio mediante inteligencia artificial ha avanzado significativamente en los últimos años, permitiendo replicar, transformar y crear contenido sonoro con alto nivel de realismo. Dentro de estas tecnologías destacan los sistemas Text-to-Speech (TTS), capaces de convertir texto en voz; Speech-to-Text (STT), orientados a la transcripción de audio; y los modelos de voice cloning, diseñados para imitar la identidad vocal de una persona. Adicionalmente, han surgido modelos capaces de generar música completa a partir de instrucciones textuales.

### 3.1 Sección 1: Clonación de voz (Voice Cloning)

En primer lugar, se evaluó la capacidad de clonación de voz utilizando una aplicación gratuita. Para el entrenamiento del modelo se utilizó una muestra de audio de aproximadamente un minuto, con un tamaño cercano a 1 MB. A pesar de la reducida cantidad de datos, el resultado obtenido fue notablemente realista.
La voz generada logró replicar con alta precisión aspectos como tono, ritmo y entonación, alcanzando un nivel de similitud que resultó indistinguible para personas cercanas, incluyendo familiares y amigos. Esto evidencia no solo el avance actual de estas herramientas, sino también su potencial riesgo, ya que un entrenamiento más amplio podría producir imitaciones prácticamente indistinguibles de la voz original.

---

### 3.2 Audio modelo y Audio original

Registro de audio sin modificación.

[Escuchar audio](https://drive.google.com/file/d/18b_-tmXc_Abuz-rtacHwuCOtrLE42plB/view?usp=sharing)

Audio generado o modificado mediante modelo de IA.

[Escuchar audio](https://drive.google.com/file/d/1pqG08NTEPX1Kz3NEoGWAG_TYhzY7OAXa/view?usp=sharing)

---

### 3.3 Audio FakeYou

Por otra parte, existen múltiples plataformas que funcionan como repositorios de voces sintéticas, ofreciendo miles de modelos para tareas como Text-to-Speech (TTS) o Voice-to-Voice (VTV). Estas herramientas permiten generar audio replicando voces de personajes, celebridades o perfiles previamente entrenados. Uno de los ejemplos más conocidos es FakeYou, plataforma que se popularizó debido a la difusión masiva de audios tipo “meme” en internet utilizando voces reconocibles.
El uso de estas herramientas resulta notablemente sencillo, ya que basta con seleccionar la modalidad y modelo deseado, para luego proporcionar el texto o audio correspondiente.

[Escuchar audio](https://drive.google.com/file/d/1DxMua5JDreN7BNpGaWUaudmkq3TUCl-r/view?usp=sharing)

Finalmente, se genera el resultado final. Como es habitual en este tipo de herramientas, las versiones premium ofrecen una calidad superior; sin embargo, muchas de estas tecnologías continúan siendo accesibles de forma gratuita.
Desde otro ángulo, a continuación se presenta como ejemplo de alta calidad un cover de la canción Golden, de la película KPOP Demon Hunters, interpretado mediante una recreación artificial de la voz del fallecido Freddie Mercury. Este tipo de contenido alcanzó gran difusión en redes sociales, evidenciando no solo nivel de realismo, sino su alcance:
https://www.youtube.com/watch?v=Kd18XGXicgo&list=RDKd18XGXicgo&start_radio=1

---

### 3.4 Sección 2: Generación de música mediante IA

En una segunda fase, se exploraron herramientas de generación musical basadas en inteligencia artificial, capaces de crear canciones completas a partir de instrucciones textuales. 
El proceso resultó notablemente sencillo: mediante un prompt se generó una pieza musical coherente, incluyendo instrumentación, ritmo y voz sintetizada. El resultado presenta una calidad percibida como genuinamente real.
Como ejemplo, se referencia una lista de reproducción generada enteramente mediante IA, la cual evidencia el nivel actual alcanzado por estas tecnologías:
https://www.youtube.com/watch?v=_XUwFjdW0LM&list=RD_XUwFjdW0LM&start_radio=1

Actualmente, existen múltiples plataformas gratuitas capaces de generar este tipo de contenido. Una de ellas es Riffusion AI, herramienta que utiliza una interfaz sencilla donde el usuario únicamente debe completar parámetros básicos, como se observa a continuación.

[Escuchar canción](https://drive.google.com/file/d/1arji4z_0NK7xuwhHvBsbdtiao1NDVa6a/view?usp=sharing)

[Escuchar canción](https://drive.google.com/file/d/1oWNGchH9uAu90rDjJSl0CjdoGxQvDmuC/view?usp=sharing)

---

## 4. Edición de imágenes con IA

La edición de imágenes mediante inteligencia artificial representa una evolución significativa frente a herramientas tradicionales como los editores manuales. Actualmente, los modelos generativos permiten modificar imágenes completas mediante instrucciones en lenguaje natural, manteniendo coherencia en iluminación, perspectiva y textura, generando resultados que en muchos casos resultan difíciles de distinguir de una edición real.

## 4.1 Sección 1: Edición mediante modelos generativos

En esta fase se evaluaron distintas herramientas de edición basadas en inteligencia artificial, utilizando modelos avanzados como Nano Banana Pro. Estas herramientas permiten modificar imágenes mediante instrucciones textuales, alterando elementos como entorno, iluminación o composición de la escena.
Los resultados obtenidos fueron altamente satisfactorios y evidenciaron que la calidad final depende en gran medida del nivel de detalle del prompt. Instrucciones más precisas producen resultados más realistas y coherentes, manteniendo consistencia visual en aspectos como sombras, proporciones e iluminación.

![Edición 1](Edicion1.png)
![Edición 2](Edicion2.png)
![Edición 3](Edicion3.png)
![Edición 4](Edicion4.png)

---

## 5. Avatares generados con IA

Los avatares humanos generados mediante inteligencia artificial representan una de las aplicaciones más avanzadas. A diferencia de técnicas como imágenes o deepfakes, los avatares permiten generar contenido audiovisual completo utilizando modelos de video, sincronización labial y, en muchos casos, clonación de voz.

## 5.1 Sección 1: Evaluación de avatares (modelo genérico vs modelo propio)

En esta fase se evaluaron dos tipos de avatares: uno basado en un modelo genérico generado por IA y otro construido a partir de datos reales de mi persona. En el primer caso, al tratarse de un avatar completamente sintético, los resultados fueron visualmente coherentes y estables, ya que el sistema no debía replicar una identidad específica, sino generar una representación artificial desde cero.

El caso más relevante fue el del avatar generado a partir de datos propios. Utilizando material audiovisual de mi persona, el sistema logró replicar tanto mi apariencia como mi voz mediante técnicas de clonación vocal. El resultado fue convincente, manteniendo coherencia en movimientos, entorno y sincronización entre voz y gestos.
Aun así, se identificaron pequeñas inconsistencias en detalles finos, especialmente en movimientos de la boca y expresiones faciales. Aunque estas imperfecciones pueden detectarse en un análisis detallado, el resultado general es lo suficientemente realista como para percibirse como auténtico en un contexto cotidiano


[![Avatar IA](CapturaAvatarIA.png)](https://drive.google.com/file/d/1q4Cit47Syr5k8JGOGdokKsqSf37htQfS/view?usp=sharing)

[![Avatar Deepfake](CapturaAvatarDeepfake.png)](https://drive.google.com/file/d/1I90-R4NeN29ZUiD5mmUvtpGWTWuG-QbG/view?usp=sharing)

---

## 6. Modelos 3D

La generación de modelos tridimensionales mediante inteligencia artificial permite recrear no solo la apariencia de un sujeto, sino también su estructura espacial. Estas herramientas utilizan técnicas de reconstrucción asistida por IA y modelos capaces de inferir geometría, textura y profundidad, generando representaciones digitales manipulables en entornos 3D [23].

## 6.1 Modelos 3D Sección 1: Generación de modelos 3D (prueba práctica)

Como parte de esta investigación, se realizó una prueba básica de generación de modelos 3D utilizando imágenes como entrada para reconstruir una representación tridimensional de mi persona, permitiendo su visualización desde distintos ángulos.
Aunque los resultados presentaron limitaciones en detalles finos y texturas, especialmente en el rostro y las manos, el nivel alcanzado fue suficiente para evidenciar el potencial de estas tecnologías. Herramientas más avanzadas o versiones premium permiten mejorar considerablemente la calidad final, ampliando los escenarios de suplantación y manipulación en entornos digitales. Para ver un impacto real de este escenario, detallar la siguiente noticia: https://www.xataka.com/seguridad/alguien-ha-impreso-cara-3d-para-intentar-burlar-reconocimiento-facial-moviles-solo-se-salva-uno

![Modelo 3D](Modelo3D.png)

---

## 7. Archivos Ofimáticos:

La generación de contenido mediante inteligencia artificial también abarca documentos ofimáticos como textos, hojas de cálculo o presentaciones. La capacidad actual de estos modelos para producir archivos con estructura y apariencia profesional introduce nuevos riesgos, especialmente en contextos académicos, empresariales o jurídicos, donde este tipo de documentos puede ser utilizado como evidencia o soporte de decisiones.

## 7.1 Sección 1: Generación de documentos (prueba práctica)

En esta fase se evaluó la capacidad de la inteligencia artificial para generar archivos ofimáticos, incluyendo texto plano, hojas de cálculo, documentos Word y presentaciones. El proceso requirió instrucciones simples y permitió obtener contenido organizado, coherente y adaptado a cada formato. Los resultados muestran que la IA puede producir documentos con apariencia legítima, aunque la calidad depende en gran medida del prompt y de las limitaciones de la herramienta utilizada. Se generaron textos formales, tablas con estructura lógica y presentaciones con jerarquía visual clara.
Esta capacidad supone un riesgo relevante, ya que estos archivos pueden utilizarse para simular registros, construir narrativas falsas o respaldar hechos inexistentes. 

![ArchivosOfimaticos](ArchivosOfimaticos.png)

Por si se quiere comprobar, los archivos se encuentran en el repositorio: 

Como: evidencias.csv -- registro
_actividad.txt -- informe_forense.docx -- ejemplo_IA.xlsx

---
## 8. Programación:

La inteligencia artificial también ha alcanzado el ámbito del desarrollo de software, permitiendo generar código funcional a partir de instrucciones en lenguaje natural. Actualmente, estos modelos pueden crear scripts, aplicaciones y sistemas completos sin requerir conocimientos avanzados de programación, introduciendo nuevos riesgos asociados a automatización, manipulación y generación.

## 8.1 Sección 1: Generación de código mediante IA

En esta fase se evaluó la capacidad de la inteligencia artificial para generar código mediante instrucciones en lenguaje natural, utilizando herramientas como ChatGPT y GitHub Copilot. Fue posible obtener scripts funcionales para tareas como análisis de archivos, automatización de procesos y generación de reportes.
Planteándolo desde la ciberseguridad, esta capacidad tiene implicaciones relevantes. Aunque facilita la creación de herramientas legítimas, también puede utilizarse para automatizar ataques, manipular información o generar artefactos que aparenten legitimidad técnica. El siguiente sencillo ejemplo demuestra este punto:

![Programacion1](Programacion1.png)

![Programacion2](Programacion2.png)

---

## 8. Conversaciones falsas (chatbots):

Los chatbots impulsados por inteligencia artificial permiten simular conversaciones naturales mediante modelos de lenguaje avanzados. En combinación con tecnologías como la clonación de voz, estos sistemas pueden replicar la forma de comunicarse de una persona específica, ampliando significativamente su potencial en escenarios tanto legítimos como de suplantación digital.

## 8.1 Sección 1: Chatbots personalizados y simulación de identidad

En esta fase se exploró la posibilidad de crear un chatbot personalizado utilizando plataformas como Character.AI, capaces de imitar el estilo de comunicación, personalidad y tono de una persona específica.
Algunas implementaciones avanzadas integran clonación de voz, permitiendo que el chatbot no solo escriba, sino también “hable” como el individuo replicado. Durante esta prueba no fue posible implementar un chatbot funcional debido a limitaciones técnicas de la cuenta utilizada. No obstante, se incluye un material audiovisual de referencia que demuestra el proceso de creación y el resultado final de este tipo de sistemas.

La Referencia en cuestión es : https://www.tiktok.com/@davidhosting/video/7377559499764272389

---
