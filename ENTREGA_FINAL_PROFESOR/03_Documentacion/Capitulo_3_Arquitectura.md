# CAPÍTULO 3: ARQUITECTURA DE INFERENCIA (DEEP LEARNING)

## 3.1. Diseño Híbrido: CNN + BiLSTM + Attention
El núcleo analítico de este trabajo reside en una arquitectura de Red Neuronal Profunda diseñada específicamente para el procesamiento de series temporales no estacionarias. El modelo adopta un diseño híbrido, combinando las fortalezas de tres paradigmas del Deep Learning:

1. **Redes Neuronales Convolucionales (CNN) 1D:** Encargadas de actuar como un "escáner de microestructura". La red ingiere tensores de tamaño `(60, 4)` que representan 60 periodos históricos de precios (Apertura, Máximo, Mínimo, Cierre) normalizados porcentualmente. Las capas convolucionales aplican filtros que detectan, de manera espacial y local, formaciones topológicas como rupturas, zonas de consolidación o contracciones de volatilidad, extrayendo *features* de bajo nivel de forma autónoma.

2. **Memoria a Corto y Largo Plazo Bidireccional (BiLSTM):** Tras la fase de extracción convolucional, los mapas de características resultantes se inyectan en capas LSTM bidireccionales. A diferencia de las redes recurrentes tradicionales, las LSTM están diseñadas para mitigar el problema del desvanecimiento del gradiente (*vanishing gradient*), reteniendo memoria de eventos lejanos en la serie temporal. La bidireccionalidad permite al modelo contextualizar un evento analizando tanto la información histórica precedente como la subsecuente dentro de la ventana de análisis, capturando así la inercia (momentum) del mercado de forma holística.

3. **Mecanismo de Atención (Attention):** El mercado financiero es asimétrico; no todas las velas de precios importan por igual. Un *crash* o una vela de absorción institucional masiva encierra mucha más información predictiva que cincuenta sesiones de baja volatilidad. Por ello, se acopló una capa de Atención que calcula dinámicamente un tensor de pesos, indicando a la red neuronal qué secuencias de la historia reciente deben priorizarse para realizar la inferencia final.

## 3.2. Proceso de Entrenamiento y Regularización
Para el entrenamiento del modelo, se utilizó el optimizador AdamW con decaimiento de peso y una función de pérdida de entropía cruzada categórica escasa (*Sparse Categorical Crossentropy*). Dada la altísima propensión de los modelos financieros al sobreajuste (*overfitting*), la arquitectura implementa una agresiva estrategia de regularización:
- **Dropout Espacial (1D):** Desactiva aleatoriamente mapas de características enteros durante la fase convolucional, forzando a los filtros a no codepender de micro-patrones específicos del conjunto de entrenamiento.
- **Batch Normalization:** Normaliza las activaciones internas de las capas neuronales, estabilizando y acelerando la convergencia en el complejo paisaje de pérdidas financieras.
- **Early Stopping:** Monitoreo estricto del *Validation Loss* durante el entrenamiento, deteniendo la iteración si el modelo comienza a memorizar ruido en lugar de aprender generalizaciones.

## 3.3. Inferencia y Umbrales de Confianza (Thresholding)
El modelo finaliza en una capa Densa con activación *Softmax*, produciendo un vector de 6 dimensiones que representa la probabilidad asignada a cada una de las *Super Labels*. El sistema no opera de forma determinista basándose en la etiqueta mayoritaria, sino que exige un rigor estadístico severo: se configuró un Umbral de Confianza (`PROB_UMBRAL = 0.93`). 

La red debe estar segura al menos en un 93% de que el patrón inminente pertenece a la clase `ALC_CONT` (Continuación Alcista Fuerte) para que el evento se considere matemáticamente digno de inversión. Este umbral elimina más del 80% de las inferencias ruidosas, transformando un sistema especulativo en un detector de francotirador estadístico.
