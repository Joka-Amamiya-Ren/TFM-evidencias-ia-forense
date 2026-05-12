# Fundamentos técnicos de IA generativa

Este apartado reúne material técnico complementario relacionado con el funcionamiento general de modelos de inteligencia artificial generativa, incluyendo redes neuronales, machine learning, deep learning, GANs y modelos de difusión.

Su finalidad es servir como apoyo conceptual al lector, sin afectar la extensión principal del documento del TFM.

El avance reciente de la inteligencia artificial generativa ha permitido la creación de contenido multimedia altamente realista mediante modelos como redes generativas adversarias (GAN) y modelos de difusión. Estas tecnologías han evolucionado hasta producir imágenes, audio y video sintético con un nivel de fidelidad que dificulta su distinción respecto al contenido auténtico, lo que ha impulsado su adopción en múltiples sectores. Sin embargo, esta misma capacidad ha dado lugar a riesgos significativos asociados a la manipulación de información, especialmente en contextos donde la veracidad del contenido es crítica.

Definiciones Previas: 
El mundo de la Inteligencia Artificial esta sostenido por distintas mecánicas tecnológicas introducidas en los últimos años, las cuales son necesarias precisar para el entendimiento posterior de las técnicas de generación y su método de detección. A continuación, una breve apertura de términos:

Redes Neuronales Artificiales (ANN): 

Las redes neuronales artificiales son modelos computacionales inspirados en la estructura y funcionamiento del cerebro humano, diseñados para identificar patrones y relaciones en grandes volúmenes de datos. Gracias a este mecanismo, las ANN son capaces de generar un “sistema adaptable que las computadoras utilizan para aprender de sus errores y mejorar continuamente. De esta forma, las redes neuronales artificiales intentan resolver problemas complicados, como la realización de resúmenes de documentos o el reconocimiento de rostros, con mayor precisión.” [10]

Estas redes están compuestas por capas de nodos interconectados que procesan información de manera jerárquica, permitiendo el aprendizaje automático a partir de ejemplos sin necesidad de reglas explícitas programadas por un ser humano en cuestión. Son la base teórica y fundamental de la Inteligencia Artificial tal y como la conocemos hoy en día.

 
https://carballar.com/wp-content/uploads/2023/03/red-neuronal-esquema-600x313-1.jpg

Nuestras redes neuronales son una función matemática compuesta por capas (Ver. Img), cuyo objetivo es aproximar una relación entre entrada y salida. El funcionamiento de las redes neuronales artificiales se basa en el ajuste continuo de sus parámetros internos, principalmente los pesos y los sesgos. 

Los pesos son valores que determinan qué tan importante es cada dato de entrada para la red. Se pueden entender como multiplicadores numéricos que “amplifican” o reducen la influencia de cada entrada, haciendo que algunos datos tengan mayor impacto que otros en el resultado final a medida que la información atraviesa las distintas capas de la red. Por otro lado, los sesgos son valores adicionales que se suman al cálculo de cada neurona. Su función es permitir que la neurona pueda generar una salida incluso cuando las entradas son muy bajas o incluso cero. Es importante aclarar que un valor de entrada en cero no significa que no existan pesos, sino que esa entrada en particular no aporta información en ese momento. El sesgo actúa como un valor constante que siempre se añade al resultado, ayudando a que la neurona no dependa únicamente de las entradas para activarse. [11]

Sin sesgo, si todas las entradas fueran cero o muy pequeñas, la neurona no se activaría. Sin embargo, en muchos casos es necesario que la neurona responda incluso ante estímulos débiles. Aquí es donde el sesgo juega un papel clave, ya que define qué tan fácil o difícil es que una neurona se active, es decir, su nivel de sensibilidad. Ahora bien, después de realizar el cálculo con pesos y sesgo, entra en juego la función de activación. Esta es una operación matemática que decide si el resultado obtenido es lo suficientemente relevante como para que la neurona “se active” y envíe información a la siguiente capa. En términos simples, evalúa si el valor calculado es útil (alto) o no (bajo). [11]

Todo el proceso anteriormente descrito es el que permite a la red neuronal modelar relaciones complejas en los datos, evitando que se limite a realizar simples operaciones lineales. Con la información previa clarificada, podemos precisar el proceso de las capas.

1.	Capa de entrada:  Cada nodo/neurona recibe un valor (llamado feature). Ejemplo: píxeles de una imagen, amplitud de audio, etc. En la capa de entrada se definen las características utilizadas para las predicciones a usar para formular el valor final. Cada característica requiere de una neurona de entrada. 	
2.	Conexiones y pesos: Al decretar el peso de cada valor, la neurona realiza el cálculo: suma ponderada = (entrada × peso) + sesgo. Aquí la red decide la importancia de cada patrón.
3.	Capas Ocultas: En esencia, el aprendizaje ocurre en esta sección. Son múltiples capas (150 en aproximación [12]) que reciben un valor, se generan grandes cantidades de cálculos basados en pesos y sesgos mientras realiza propagaciones por cada una de las capas, disparando funciones de activación y modelaciones complejas.
4.	Capa de salida: Entrega el resultado final. Una neurona de salida por cada dato de salida.

Propagación hacia adelante (Forward Propagation): Es el proceso en el que los datos de entrada pasan por la red neuronal desde la primera capa hasta la última. En cada neurona, los valores se multiplican por sus pesos, se suman, se les añade el sesgo y se aplica una función de activación. Este proceso se repite capa por capa hasta obtener una salida final, que corresponde a la predicción de la red. Mas detalles aquí: [13]

Propagación hacia atrás (Backward Propagation): Una vez obtenida la predicción, esta se compara con el valor real para calcular el error. La propagación hacia atrás consiste en usar ese error para ajustar los pesos y sesgos de la red, empezando desde la capa de salida hacia atrás. El objetivo es reducir el error en futuras predicciones, permitiendo que la red aprenda progresivamente. Mas detalles aquí: [14]

Durante el entrenamiento, la red compara sus resultados con los valores reales y calcula un error. Para reducir este error, utiliza un algoritmo llamado descenso de gradiente, que ajusta gradualmente los pesos y sesgos en la dirección que mejora las predicciones. Este proceso se repite múltiples veces hasta que la red logra un buen nivel de precisión.

A medida que procesa datos, la red aprende a identificar patrones ocultos (relaciones que no son evidentes a simple vista), lo que le permite realizar tareas como clasificar imágenes o generar contenido. Su estructura, conocida como arquitectura, puede variar desde modelos simples hasta redes profundas con muchas capas, lo que aumenta su capacidad para representar información compleja y explica por qué estas técnicas son la base de sistemas avanzados de inteligencia artificial.


Machine Learning

El aprendizaje automático (Machine Learning) es una rama de la inteligencia artificial que permite a los sistemas aprender a partir de datos sin necesidad de ser programados con reglas explícitas. En lugar de indicar paso a paso qué hacer, se entrena al sistema con ejemplos para que identifique patrones y relaciones en la información. Este aprendizaje se basa en modelos que ajustan sus parámetros —como los pesos y sesgos previamente definidos— para mejorar sus resultados con el tiempo, utilizando procesos como la propagación hacia adelante (generar una predicción) y la propagación hacia atrás (corregir el error). [15]

En el contexto de este trabajo, el Machine Learning constituye la base tecnológica que permite a los sistemas analizar grandes volúmenes de datos multimedia y aprender sus características. Esto es fundamental tanto para la generación como para la detección de contenido sintético, ya que los mismos principios que permiten identificar patrones reales pueden ser utilizados para replicarlos o para encontrar inconsistencias en ellos.

Deep Learning

El aprendizaje profundo (Deep Learning) es una subdisciplina del Machine Learning que utiliza redes neuronales artificiales con múltiples capas para aprender representaciones más complejas de los datos. A diferencia de enfoques más simples, estas redes procesan la información en varias etapas, donde cada capa extrae características cada vez más abstractas, como bordes, formas o incluso estructuras completas en imágenes y audio. Este proceso se apoya en los mecanismos ya descritos, como el uso de pesos, sesgos y funciones de activación. [16]

Su relevancia en este trabajo es central, ya que las técnicas modernas de generación de contenido como imágenes, audio o video manipulados se basan principalmente en modelos de Deep Learning. Estas mismas capacidades que permiten generar contenido altamente realista son las que dificultan su detección, lo que justifica la necesidad de desarrollar métodos forenses capaces de identificar patrones artificiales, inconsistencias o rastros dejados por estos modelos.


Modelos Generativos.
Los modelos generativos son sistemas de inteligencia artificial diseñados para crear nuevos datos a partir de patrones aprendidos durante el entrenamiento. A diferencia de los modelos discriminativos, que se enfocan en clasificar o predecir, los modelos generativos buscan aproximar la distribución de los datos originales para producir contenido sintético, como imágenes, audio o video, con características similares a los datos reales

Los modelos generativos son el núcleo del problema, ya que son los responsables de la creación de contenido multimedia altamente realista, como deepfakes o audio clonado. Su capacidad para imitar patrones reales con gran precisión hace que la distinción entre contenido auténtico y manipulado sea cada vez más difícil, lo que plantea desafíos directos para el análisis forense y la valoración de evidencia digital. Por ej., cito este video, que fue famoso por engañar a una buena parte de la población en internet.

 

GAN (Generative Adversarial Network): 
 
https://www.linkedin.com/pulse/exploring-fascinating-realm-generative-adversarial-networks-kaurav/

Las redes generativas adversarias (GAN) son un tipo de modelo generativo compuesto por dos redes neuronales que compiten entre sí: un generador, que crea contenido sintético, y un discriminador, que intenta distinguir entre contenido real y falso. Durante el entrenamiento, el generador mejora su capacidad de crear datos realistas, mientras que el discriminador se vuelve más preciso en detectarlos, en un proceso iterativo basado en propagación hacia adelante y hacia atrás.

Las GAN son especialmente relevantes porque han sido una de las principales tecnologías utilizadas en la creación de deepfakes, particularmente en la manipulación de rostros y video. Este enfoque adversarial permite generar contenido con un alto nivel de realismo, pero también introduce ciertos patrones o artefactos que pueden ser analizados desde un enfoque forense. El hecho de que constantemente la IA se desafié antes de entregar un producto final nos muestra que, a ojos de una Inteligencia Artificial, la imagen generada es en efecto, real. El ejemplo más mediático y famoso fue Drag your GAN, un proyecto de edición en tiempo real basado enteramente en IA con redes GAN. El repositorio lo podemos encontrar referenciado en la siguiente imagen, que muestra el potencial de una red de esta clase, cuya tecnología es considerada “obsoleta” actualmente.

 
https://github.com/XingangPan/DragGAN

Modelos de Difusión

 
 

https://aurorasolar-com.translate.goog/blog/putting-ai-to-the-test-generative-adversarial-networks-vs-diffusion-models/?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc

Los modelos de difusión son una clase más reciente de modelos generativos que crean datos a partir de un proceso progresivo de eliminación de ruido. Inicialmente, parten de una señal completamente aleatoria (ruido) y, mediante múltiples iteraciones, van reconstruyendo una imagen o señal coherente basándose en lo aprendido durante el entrenamiento. Este proceso inverso está guiado por redes neuronales profundas que predicen cómo debe “limpiarse” el ruido en cada paso.

Estos modelos de difusión representan el estado del arte actual (o al menos, más usado) en generación de imágenes realistas, siendo utilizados por herramientas modernas capaces de producir contenido difícil de distinguir del real. Su funcionamiento introduce nuevos desafíos para la detección forense, ya que generan menos artefactos evidentes en comparación con las GAN, lo que exige técnicas de análisis más avanzadas.

GAN vs Modelos de Difusión [17]:

CRITERIO	GAN (GENERATIVE ADVERSARIAL NETWORKS)	MODELOS DE DIFUSIÓN
ARQUITECTURA	Dos redes: generador vs discriminador (competencia)	Un solo modelo que elimina ruido progresivamente
MÉTODO DE ENTRENAMIENTO	Entrenamiento adversarial (inestable, requiere balance)	Proceso iterativo de ruido → limpieza (más estable) 
VELOCIDAD DE GENERACIÓN	Muy rápida (casi en tiempo real)	Lenta (requiere múltiples pasos) 
CALIDAD DE SALIDA	Alta calidad visual, pero puede repetir patrones	Muy alta calidad y mayor nivel de detalle 
DIVERSIDAD DE RESULTADOS	Limitada (riesgo de mode collapse)	Alta diversidad de muestras 
ESTABILIDAD EN ENTRENAMIENTO	Baja (difícil de entrenar)	Alta (más robusto) 
EFICIENCIA DE DATOS	Más eficiente con pocos datos	Menos eficiente, requiere más datos 
COSTO COMPUTACIONAL	Alto en entrenamiento, bajo en generación	Muy alto (entrenamiento y generación) 
CAPACIDAD PARA DATOS COMPLEJOS	Buena, pero limitada en alta complejidad	Excelente en datos complejos y alta dimensionalidad 
EJEMPLOS	Drag Gan, StyleGan, DeepFaceLab	Stable Diffusion, DALL-E, Midjourney

Es evidenciable que tanto las redes generativas adversarias como los modelos de difusión han alcanzado un nivel de madurez suficiente para generar contenido multimedia altamente realista, aunque mediante enfoques técnicos distintos. Mientras que las GAN destacan por su velocidad y han sido ampliamente utilizadas en la creación de deepfakes tradicionales, los modelos de difusión representan una evolución en términos de calidad, estabilidad y capacidad para reproducir distribuciones complejas de datos. Esta transición tecnológica implica que los rastros y artefactos que antes eran identificables en modelos más antiguos (Ej. Cuando mirábamos el número de dedos o las manos [18]) tienden a reducirse o desaparecer en las nuevas arquitecturas.


