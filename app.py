import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

st.set_page_config(
    page_title="VAE Generativo — MNIST",
    page_icon="🔢",
    layout="wide"
)

# ── CSS mínimo ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; }
    h1 { font-size: 1.4rem !important; }
    h3 { font-size: 1rem !important; color: #888; font-weight: 400 !important; }
    .metric-card {
        background: #f8f8f8;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 22px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Importar TF con aviso ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando TensorFlow...")
def load_tf():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    return tf

tf = load_tf()


# ── Modelo VAE ───────────────────────────────────────────────────────────────
class Sampling(tf.keras.layers.Layer):
    """Reparametrization trick: z = mu + sigma * epsilon"""
    def call(self, inputs):
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim   = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * log_var) * epsilon


def build_encoder(latent_dim):
    inputs  = tf.keras.Input(shape=(784,), name="encoder_input")
    x       = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x       = tf.keras.layers.Dense(256, activation="relu")(x)
    mu      = tf.keras.layers.Dense(latent_dim, name="mu")(x)
    log_var = tf.keras.layers.Dense(latent_dim, name="log_var")(x)
    z       = Sampling(name="z")([mu, log_var])
    return tf.keras.Model(inputs, [mu, log_var, z], name="encoder")


def build_decoder(latent_dim):
    inputs = tf.keras.Input(shape=(latent_dim,), name="decoder_input")
    x      = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x      = tf.keras.layers.Dense(512, activation="relu")(x)
    output = tf.keras.layers.Dense(784, activation="sigmoid")(x)
    return tf.keras.Model(inputs, output, name="decoder")


class VAE(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder    = build_encoder(latent_dim)
        self.decoder    = build_decoder(latent_dim)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker    = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=-1
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
            )
            total_loss = recon_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


# ── Datos MNIST ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando MNIST...")
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


# ── Entrenamiento con callback para Streamlit ────────────────────────────────
def train_vae(latent_dim, epochs, batch_size):
    x_train, y_train, x_test, y_test = load_mnist()

    vae = VAE(latent_dim=latent_dim)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    history = {"total_loss": [], "recon_loss": [], "kl_loss": []}

    progress_bar  = st.progress(0, text="Iniciando entrenamiento...")
    loss_placeholder = st.empty()
    chart_placeholder = st.empty()

    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            history["total_loss"].append(float(logs.get("total_loss", 0)))
            history["recon_loss"].append(float(logs.get("recon_loss", 0)))
            history["kl_loss"].append(float(logs.get("kl_loss", 0)))

            pct  = (epoch + 1) / epochs
            progress_bar.progress(pct, text=f"Época {epoch+1}/{epochs}")

            col1, col2, col3 = loss_placeholder.columns(3)
            col1.metric("Loss total",    f"{history['total_loss'][-1]:.1f}")
            col2.metric("Recon loss",    f"{history['recon_loss'][-1]:.1f}")
            col3.metric("KL loss",       f"{history['kl_loss'][-1]:.2f}")

            if len(history["total_loss"]) > 1:
                fig, ax = plt.subplots(figsize=(6, 2.5))
                ax.plot(history["total_loss"], label="Total",       color="#534AB7")
                ax.plot(history["recon_loss"], label="Reconstrucc.", color="#1D9E75", linestyle="--")
                ax.plot(history["kl_loss"],    label="KL",          color="#D85A30", linestyle=":")
                ax.set_xlabel("Época", fontsize=10)
                ax.set_ylabel("Loss",  fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close(fig)

    vae.fit(
        x_train, epochs=epochs, batch_size=batch_size,
        shuffle=True, verbose=0,
        callbacks=[StreamlitCallback()]
    )

    progress_bar.empty()
    return vae, x_test, y_test, history


# ── Helpers de visualización ─────────────────────────────────────────────────
def img_from_z(decoder, z_vector):
    z = np.array([z_vector], dtype="float32")
    pixel = decoder.predict(z, verbose=0)[0].reshape(28, 28)
    return pixel


def scatter_latente(encoder, x_test, y_test, latent_dim):
    mu, _, _ = encoder.predict(x_test[:3000], verbose=0, batch_size=256)

    if latent_dim == 2:
        x_vals, y_vals = mu[:, 0], mu[:, 1]
        xlabel, ylabel = "z₁", "z₂"
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        mu2 = pca.fit_transform(mu)
        x_vals, y_vals = mu2[:, 0], mu2[:, 1]
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)"

    cmap   = plt.cm.get_cmap("tab10", 10)
    fig, ax = plt.subplots(figsize=(7, 5))
    for digit in range(10):
        mask = y_test[:3000] == digit
        ax.scatter(x_vals[mask], y_vals[mask], c=[cmap(digit)],
                   s=8, alpha=0.7, label=str(digit))
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(title="Dígito", markerscale=2, fontsize=9,
              loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def interpolation_grid(decoder, z_a, z_b, steps=10):
    alphas  = np.linspace(0, 1, steps)
    images  = [img_from_z(decoder, (1-a)*z_a + a*z_b) for a in alphas]
    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.2, 1.4))
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout(pad=0.3)
    return fig


def random_grid(decoder, latent_dim, n=16):
    z = np.random.randn(n, latent_dim).astype("float32")
    imgs = decoder.predict(z, verbose=0).reshape(n, 28, 28)
    cols = 8
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    for ax, img in zip(axes.flatten(), imgs):
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout(pad=0.3)
    return fig


def latent_grid_2d(decoder, rng=3.0, steps=15):
    """Cuadrícula navegando el espacio latente 2D"""
    zs = np.linspace(-rng, rng, steps)
    fig, axes = plt.subplots(steps, steps, figsize=(steps * 0.8, steps * 0.8))
    for i, z2 in enumerate(reversed(zs)):
        for j, z1 in enumerate(zs):
            img = img_from_z(decoder, [z1, z2])
            axes[i][j].imshow(img, cmap="gray_r", vmin=0, vmax=1)
            axes[i][j].axis("off")
    plt.tight_layout(pad=0.1)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  UI PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

st.title("VAE Generativo — MNIST")
st.markdown("### Variational Autoencoder interactivo para explorar y generar dígitos")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuración del modelo")

    latent_dim = st.select_slider(
        "Dimensión latente",
        options=[2, 4, 8, 16, 32],
        value=2,
        help="Con dim=2 puedes navegar la cuadrícula 2D del espacio latente"
    )

    epochs = st.slider("Épocas de entrenamiento", 5, 50, 20, step=5)
    batch_size = st.select_slider("Batch size", options=[64, 128, 256], value=128)

    entrenar = st.button("Entrenar modelo", type="primary", use_container_width=True)

    st.divider()
    st.caption("Tip: con `latent_dim=2` se activa la cuadrícula 2D del espacio latente, muy visual para clase.")

    if "history" in st.session_state:
        h = st.session_state["history"]
        st.subheader("Última sesión")
        c1, c2 = st.columns(2)
        c1.metric("Loss final", f"{h['total_loss'][-1]:.1f}")
        c2.metric("KL final",   f"{h['kl_loss'][-1]:.2f}")


# ── Entrenamiento ────────────────────────────────────────────────────────────
if entrenar:
    with st.spinner(""):
        st.info(f"Entrenando VAE con dim_latente={latent_dim}, {epochs} épocas, batch={batch_size}…")
        vae, x_test, y_test, history = train_vae(latent_dim, epochs, batch_size)
        st.session_state["vae"]        = vae
        st.session_state["x_test"]     = x_test
        st.session_state["y_test"]     = y_test
        st.session_state["latent_dim"] = latent_dim
        st.session_state["history"]    = history
    st.success("Entrenamiento completado.")


# ── Tabs de exploración ──────────────────────────────────────────────────────
if "vae" not in st.session_state:
    st.info("Configura los parámetros en el panel izquierdo y presiona **Entrenar modelo** para comenzar.")
    st.stop()

vae        = st.session_state["vae"]
x_test     = st.session_state["x_test"]
y_test     = st.session_state["y_test"]
ld         = st.session_state["latent_dim"]

encoder = vae.encoder
decoder = vae.decoder

tabs = st.tabs([
    "Generar dígitos",
    "Espacio latente 2D" if ld == 2 else "Espacio latente (PCA)",
    "Interpolación",
    "Muestras aleatorias",
    "Cuadrícula latente 2D" if ld == 2 else "Cuadrícula (solo dim=2)",
])

# ─ Tab 1: Generar con sliders ────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Mover el vector latente z")
    st.caption("Cada slider controla una dimensión del espacio latente. La imagen se regenera al instante.")

    cols_sliders = st.columns(min(ld, 4))
    z_vector = []
    for i in range(ld):
        col = cols_sliders[i % 4] if ld > 4 else cols_sliders[i]
        val = col.slider(f"z{i+1}", -3.0, 3.0, 0.0, step=0.05, key=f"z_{i}")
        z_vector.append(val)

    img = img_from_z(decoder, z_vector)

    col_img, col_info = st.columns([1, 2])
    with col_img:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    with col_info:
        st.markdown("**Vector z actual:**")
        st.code(f"z = {[round(v, 2) for v in z_vector]}")
        st.markdown("**Cómo funciona:**")
        st.markdown(
            "El decoder toma el vector `z` y lo transforma en una imagen de 28×28 píxeles. "
            "Al variar las dimensiones de `z` navegas el espacio aprendido por el VAE."
        )

# ─ Tab 2: Scatter espacio latente ────────────────────────────────────────────
with tabs[1]:
    st.subheader("Distribución del espacio latente" + (" 2D" if ld == 2 else " (proyección PCA)"))
    st.caption("Cada punto es un dígito del test set codificado en su μ (media del encoder). El color indica la clase.")

    with st.spinner("Generando scatter..."):
        fig = scatter_latente(encoder, x_test, y_test, ld)
    st.pyplot(fig)
    plt.close(fig)

    if ld > 2:
        st.info("Con `latent_dim > 2` se usa PCA para proyectar en 2D. "
                "Entrena con `latent_dim = 2` para ver el espacio directamente.")

# ─ Tab 3: Interpolación ──────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Interpolación entre dos dígitos")
    st.caption("El VAE puede interpolar suavemente entre dos puntos del espacio latente.")

    c1, c2, c3 = st.columns(3)
    digit_a = c1.selectbox("Dígito origen", list(range(10)), index=3, key="da")
    digit_b = c2.selectbox("Dígito destino", list(range(10)), index=8, key="db")
    steps   = c3.slider("Pasos", 5, 20, 10, key="interp_steps")

    if st.button("Generar interpolación", key="btn_interp"):
        idx_a = np.where(y_test == digit_a)[0][0]
        idx_b = np.where(y_test == digit_b)[0][0]

        mu_a, _, _ = encoder.predict(x_test[[idx_a]], verbose=0)
        mu_b, _, _ = encoder.predict(x_test[[idx_b]], verbose=0)

        fig = interpolation_grid(decoder, mu_a[0], mu_b[0], steps=steps)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Interpolación lineal de {steps} pasos entre el dígito {digit_a} y el {digit_b} en el espacio latente.")

# ─ Tab 4: Muestras aleatorias ────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Muestras aleatorias del espacio latente")
    st.caption("Se muestrean vectores z ~ N(0,1) y el decoder los convierte en imágenes.")

    n_samples = st.slider("Número de imágenes", 8, 32, 16, step=8, key="n_rand")

    if st.button("Generar muestras", key="btn_rand"):
        fig = random_grid(decoder, ld, n=n_samples)
        st.pyplot(fig)
        plt.close(fig)

# ─ Tab 5: Cuadrícula 2D ──────────────────────────────────────────────────────
with tabs[4]:
    if ld == 2:
        st.subheader("Cuadrícula del espacio latente 2D")
        st.caption("Navega sistemáticamente el espacio latente variando z₁ (eje X) y z₂ (eje Y).")

        col_r, col_s = st.columns(2)
        rango = col_r.slider("Rango [-r, r]", 1.0, 4.0, 3.0, step=0.5, key="rng")
        pasos = col_s.slider("Pasos por eje", 10, 20, 15, step=1, key="pasos")

        if st.button("Generar cuadrícula", key="btn_grid"):
            with st.spinner("Generando cuadrícula..."):
                fig = latent_grid_2d(decoder, rng=rango, steps=pasos)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Cuadrícula de {pasos}×{pasos} = {pasos**2} imágenes generadas.")
    else:
        st.info("La cuadrícula 2D solo está disponible con `latent_dim = 2`. "
                "Entrena el modelo con esa configuración para activar esta vista.")
