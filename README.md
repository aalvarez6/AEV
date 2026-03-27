# VAE Generativo — MNIST

App interactiva en Streamlit para explorar un **Variational Autoencoder (VAE)** entrenado sobre el dataset MNIST.

## Características

| Tab | Descripción |
|-----|-------------|
| **Generar dígitos** | Sliders para cada dimensión de `z`. La imagen se regenera al instante |
| **Espacio latente** | Scatter plot de los dígitos del test set proyectados en `μ` (coloreados por clase) |
| **Interpolación** | Interpola suavemente entre dos dígitos en el espacio latente |
| **Muestras aleatorias** | Muestrea `z ~ N(0,1)` y genera imágenes sintéticas |
| **Cuadrícula 2D** | Navega el espacio latente completo en una grilla (solo con `latent_dim=2`) |

## Ejecutar localmente

```bash
# 1. Clonar el repo
git clone <tu-repo>
cd vae_app

# 2. Instalar dependencias (recomendado: entorno virtual)
pip install -r requirements.txt

# 3. Lanzar
streamlit run app.py
```

## Deploy en Streamlit Cloud

1. Sube este repositorio a GitHub (público o privado)
2. Ve a [share.streamlit.io](https://share.streamlit.io) y conecta tu cuenta de GitHub
3. Selecciona el repo y el archivo `app.py`
4. Haz clic en **Deploy** — Streamlit Cloud instala las dependencias automáticamente

> **Nota:** El primer deploy tarda ~5 min porque descarga TensorFlow. Las siguientes cargas son inmediatas gracias al cache.

## Parámetros recomendados para clase

| Escenario | `latent_dim` | Épocas | Resultado |
|-----------|-------------|--------|-----------|
| Demo rápida | 2 | 10 | Activa la cuadrícula 2D, entrena en ~3 min |
| Calidad media | 2 | 30 | Buena separación de clases en el scatter |
| Exploración avanzada | 8–16 | 30 | Mejor calidad de imagen, scatter con PCA |

## Arquitectura del VAE

```
Input (784)
    │
  Dense 512 → Dense 256
    │
  ┌─────┬─────────┐
  μ     log σ²    │
  └──┬──┘         │
     │  ε~N(0,1)  │
     z = μ + σ·ε  │
     │             │
  Dense 256 → Dense 512 → sigmoid
    │
  Output (784)
    │
Loss = BCE(x, x̂) + KL(N(μ,σ²) || N(0,1))
```

## Estructura del proyecto

```
vae_app/
├── app.py           # App principal
├── requirements.txt # Dependencias para Streamlit Cloud
└── README.md
```
