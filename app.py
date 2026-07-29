import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from datetime import datetime

icon = Image.open("N (2).png")

st.set_page_config(
    page_title="NeuroScan",
    page_icon=icon,
    layout="wide"
)

model = tf.keras.models.load_model("brain_tumor_model.keras")

# Títol
st.image(icon, width=90)
st.title("NeuroScan")
st.subheader("Detecció automàtica de tumors cerebrals amb Intel·ligència Artificial")

st.divider()

uploaded_file = st.file_uploader(
    "Puja una ressonància magnètica",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    img = image.resize((100,100))
    img = np.array(img)/255.0
    img = img.reshape(1,100,100,1)

    with st.spinner("Analitzant la ressonància..."):
        inici = time.time()
        time.sleep(1.5)
        prediction = model.predict(img, verbose=0)[0][0]
        temps = time.time() - inici

    col1, col2 = st.columns([1,1])


    with col1:

        st.image(
        image,
        caption="Ressonància magnètica",
        use_container_width=True
        )

    with col2:

        st.subheader("Predicció")

        if prediction >= 0.4:
            st.error("🔴 Tumor detectat")
        else:
            st.success("🟢 Sense tumor")

        st.write("### Probabilitat de tumor")

        st.progress(float(prediction))

        st.write(f"**{prediction*100:.2f}%**")

        # Confiança del model (independentment de la classe)
        confidence = max(prediction, 1 - prediction)

        st.write(f"**Confiança del model:** {confidence*100:.2f}%")

        st.markdown("### 🩺 Interpretació")

        if prediction >= 0.4:
            st.info(

                "El model prediu que aquesta ressonància és compatible amb la presència d'un tumor cerebral."
            )
        else:
            st.info(

                "El model prediu que aquesta ressonància no presenta característiques compatibles amb un tumor cerebral."
            )

        informe = f"""
            INFORME DE PREDICCIÓ

            NeuroScan v1.0

            Data: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

            ----------------------------------------

            Resultat:
            {"Tumor detectat" if prediction >= 0.4 else "Sense tumor"}

            Probabilitat de tumor:
            {prediction*100:.2f} %

            Confiança del model:
            {confidence*100:.2f} %

            Temps d'anàlisi:
            {temps:.3f} segons

            ----------------------------------------

            Aquest informe ha estat generat automàticament per NeuroScan AI.

            Aplicació desenvolupada exclusivament amb finalitats acadèmiques.

            No substitueix el diagnòstic d'un professional sanitari.
            """

        st.download_button(
            label="📄 Genera l'informe clínic (.txt)",
            data=informe,
            file_name="Informe_NeuroScanAI.txt",
            mime="text/plain"
            )

        st.write(f"**Temps d'anàlisi:** {temps:.3f} segons")



st.divider()

with st.expander("🔵 Informació tècnica del model: v1.0"):
    st.markdown("""
    - Xarxa neuronal convolucional (CNN)
    - Entrenada amb **8.764** ressonàncies magnètiques
    - Resolució d'entrada: **100 × 100 píxels**
    - Llindar de classificació: **0,40**
    - Exactitud obtinguda en el conjunt de prova: **97,5%**
    """)

with st.expander("🔐Privacitat de les dades"):
    st.markdown("""
    **Respectem la teva privacitat**

    Les imatges que puges a aquesta plataforma es processen en temps real perquè el model fagi la predicció.  

    - No s'emmagatzema cap imatge.
    - No es registra cap dada del pacient.
    - No s'envia informació a cap servidor extern.
    - Les dades s'eliminen automàticament de la memòria un cop finalitzada l'anàlisi.
    """)

st.warning(
    "⚠️ Aquesta aplicació és una demostració desenvolupada exclusivament "
    "amb finalitats acadèmiques. "
    "No substitueix el diagnòstic d'un professional sanitari."
)