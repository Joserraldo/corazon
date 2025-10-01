# Modelo IA para Predicción Cardiaca

Aplicación en **Streamlit** que predice si un paciente sufrirá o no del corazón usando un modelo **SVC** previamente entrenado y guardado con `joblib`.

## 📂 Archivos del proyecto
- `app.py` → Código principal de la aplicación Streamlit.
- `svc_model.jb` → Modelo entrenado (SVC).
- `scaler.jb` → Escalador MinMaxScaler usado en el preprocesamiento.
- `requirements.txt` → Dependencias necesarias.

## ⚙️ Instalación
1. Clonar o descargar este repositorio.
2. Crear un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate   # En Linux/Mac
   venv\Scripts\activate      # En Windows
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Ejecución
En la terminal, dentro del directorio del proyecto, ejecutar:
```bash
streamlit run app.py
```

## 🧾 Uso
1. Ajustar los valores de **Edad** y **Colesterol** con los sliders.
2. Presionar **Predecir**.
3. El modelo mostrará:
   - **0 → No sufrirá del corazón** (con imagen de corazón sano).
   - **1 → Sufrirá del corazón** (con imagen de problema cardiaco).

## 📊 Notas
- El modelo fue entrenado con las variables:
  - `edad` (25–80 años)
  - `colesterol` (120–600 mg/dL)
- Los datos fueron normalizados usando **MinMaxScaler**.
- El comportamiento de la aplicación depende de cómo se entrenó el modelo.

## 👨‍💻 Autor
Elaborado por **Jose Alejandro ® UNAB 2025**
