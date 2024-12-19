import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
import platform
import psutil
import gc
import tensorflow.keras as keras
import glob  
from visualizaciones import VisualizacionMicroclima  # Añadir esta línea

# Configuración para reducir mensajes de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Configuración para usar CPU eficientemente
physical_devices = tf.config.list_physical_devices('CPU')
try:
    # Configurar para usar toda la memoria disponible
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=None)]
    )
except RuntimeError as e:
    print(f"Error en configuración de dispositivo: {e}")

# Configurar paralelismo
tf.config.threading.set_intra_op_parallelism_threads(28)
tf.config.threading.set_inter_op_parallelism_threads(28)

# Habilitar optimizaciones
tf.config.optimizer.set_jit(True)  # Habilitar XLA
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Configurar política de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def print_system_info():
    """Imprime información detallada del sistema"""
    print("\n=== Información del Sistema ===")
    print(f"Sistema Operativo: {platform.system()} {platform.version()}")
    print(f"Procesador: {platform.processor()}")
    print(f"Núcleos Físicos: {psutil.cpu_count(logical=False)}")
    print(f"Núcleos Totales: {psutil.cpu_count()}")
    
    # Memoria RAM
    ram = psutil.virtual_memory()
    print("\n=== Memoria RAM ===")
    print(f"Total: {ram.total / (1024 ** 3):.2f} GB")
    print(f"Disponible: {ram.available / (1024 ** 3):.2f} GB")
    print(f"Usada: {ram.used / (1024 ** 3):.2f} GB")
    print(f"Porcentaje usado: {ram.percent}%")

    # Espacio en disco
    disk = psutil.disk_usage('/')
    print("\n=== Espacio en Disco ===")
    print(f"Total: {disk.total / (1024 ** 3):.2f} GB")
    print(f"Disponible: {disk.free / (1024 ** 3):.2f} GB")
    print(f"Usado: {disk.used / (1024 ** 3):.2f} GB")
    print(f"Porcentaje usado: {disk.percent}%")

# Inicialización de información del sistema
print_system_info()

class PrediccionMicroclima:
    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, df, scalers, label_encoder, num_categorias, batch_size=32, ventana_tiempo=12):
            self.df = df
            self.batch_size = batch_size
            self.ventana_tiempo = ventana_tiempo
            self.scalers = scalers
            self.label_encoder = label_encoder
            self.num_categorias = num_categorias
            self.variables_numericas = ['temperatura_C', 'humedad_relativa', 
                                      'precipitacion_mm', 'cobertura_nubes_octas', 
                                      'velocidad_viento_kmh', 'radiacion_solar_J_m2']
            self.indices = np.arange(len(df) - ventana_tiempo - 72 + 1)
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.indices) / self.batch_size))

        def __getitem__(self, idx):
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx]

            X_batch = []
            y_batch = []

            for i in batch_indices:
                # Preparar ventana de entrada
                ventana = self.df.iloc[i:i+self.ventana_tiempo][self.variables_numericas].values
                # Normalizar datos
                ventana_norm = np.zeros_like(ventana)
                for j, var in enumerate(self.variables_numericas):
                    ventana_norm[:, j] = self.scalers[var].transform(ventana[:, j].reshape(-1, 1)).ravel()
                X_batch.append(ventana_norm)

                # Preparar etiquetas
                y_seq = self.df['categoria_numerica'].iloc[i+self.ventana_tiempo:i+self.ventana_tiempo+72].values
                y_seq_onehot = np.zeros((72, self.num_categorias))
                for t, cat in enumerate(y_seq):
                    y_seq_onehot[t, cat] = 1
                y_batch.append(y_seq_onehot)

            return np.array(X_batch), np.array(y_batch)

        def on_epoch_end(self):
            np.random.shuffle(self.indices)
    class OptimizedDataGenerator(tf.keras.utils.Sequence):
        """Generador de datos optimizado para entrenamiento por lotes"""
        
        def __init__(self, X, y, batch_size=64, shuffle=True, augment=False):
            """
            Inicializa el generador.
            
            Args:
                X: Datos de entrada
                y: Etiquetas
                batch_size: Tamaño del lote
                shuffle: Si se mezclan los datos en cada época
                augment: Si se aplica aumento de datos
            """
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = augment
            self.indices = np.arange(len(self.X))
            self.on_epoch_end()
            
        def __len__(self):
            """Número de lotes por época"""
            return int(np.ceil(len(self.X) / self.batch_size))
            
        def __getitem__(self, idx):
            """Obtiene un lote de datos"""
            # Obtener índices para este lote
            batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            
            # Extraer datos del lote
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            
            # Aplicar aumento de datos si está activado
            if self.augment:
                X_batch = self._augment_batch(X_batch)
            
            return X_batch, y_batch
        
        def _augment_batch(self, X_batch):
            """Aplica técnicas de aumento de datos"""
            augmented_batch = X_batch.copy()
            
            # Añadir ruido gaussiano
            noise = np.random.normal(0, 0.01, X_batch.shape)
            augmented_batch += noise
            
            # Escalar aleatoriamente
            scale_factor = np.random.uniform(0.95, 1.05)
            augmented_batch *= scale_factor
            
            return augmented_batch
            
        def on_epoch_end(self):
            """Llamado al final de cada época"""
            if self.shuffle:
                np.random.shuffle(self.indices)
                
        def get_config(self):
            """Obtiene la configuración del generador"""
            return {
                'batch_size': self.batch_size,
                'shuffle': self.shuffle,
                'augment': self.augment
            }
        
        def save_state(self, filepath):
            """Guarda el estado del generador"""
            try:
                state = {
                    'indices': self.indices,
                    'config': self.get_config()
                }
                np.save(filepath, state)
                return True
            except Exception as e:
                print(f"Error al guardar estado: {str(e)}")
                return False
        
        def load_state(self, filepath):
            """Carga el estado del generador"""
            try:
                state = np.load(filepath, allow_pickle=True).item()
                self.indices = state['indices']
                config = state['config']
                self.batch_size = config['batch_size']
                self.shuffle = config['shuffle']
                self.augment = config['augment']
                return True
            except Exception as e:
                print(f"Error al cargar estado: {str(e)}")
                return False
    ###########################################
    def __init__(self):
        """Inicializa el modelo de predicción con configuraciones optimizadas"""
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.categorias = None
        self.num_categorias = None
        self.modelo_path = 'modelo_microclima.keras'
        self.visualizador = VisualizacionMicroclima()  # Añadir esta línea
        # Scalers individuales para cada variable
        self.scalers = {
            'temperatura_C': MinMaxScaler(),
            'humedad_relativa': MinMaxScaler(),
            'precipitacion_mm': MinMaxScaler(),
            'cobertura_nubes_octas': MinMaxScaler(),
            'velocidad_viento_kmh': MinMaxScaler(),
            'radiacion_solar_J_m2': MinMaxScaler()
        }

        # Configuración de hiperparámetros optimizados
        self.BATCH_SIZE = 128  # Aumentado para mejor rendimiento
        self.SHUFFLE_BUFFER = 10000
        self.LEARNING_RATE = 0.001
        self.WARMUP_EPOCHS = 50
        
        # Configuración de memoria
        self.CHUNK_SIZE = 1000  # Tamaño de chunk para procesamiento
        self.MAX_MEMORY_GB = 28  # Límite de memoria en GB
    def cargar_datos(self, ruta_archivo, fecha_inicio=None, fecha_fin=None):
        """Carga y preprocesa los datos con manejo de memoria optimizado"""
        try:
            # Usar read_csv con chunks para grandes datasets
            chunks = []
            for chunk in pd.read_csv(ruta_archivo, chunksize=self.CHUNK_SIZE):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            
            # Convertir fechas y establecer índice
            df['fecha'] = pd.to_datetime(df['fecha'])
            df.set_index('fecha', inplace=True)
            
            # Filtrar por fecha si se especifica
            if fecha_inicio and fecha_fin:
                mask = (df.index >= fecha_inicio) & (df.index <= fecha_fin)
                df = df[mask]
            
            # Manejar valores faltantes
            if df.isnull().any().any():
                df = self.manejar_valores_faltantes(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error al cargar los datos: {str(e)}")

    def manejar_valores_faltantes(self, df):
        """Maneja los valores faltantes en el dataset de manera eficiente"""
        variables_numericas = ['temperatura_C', 'humedad_relativa', 'precipitacion_mm',
                             'cobertura_nubes_octas', 'velocidad_viento_kmh', 
                             'radiacion_solar_J_m2']
        
        # Interpolar valores faltantes por chunks
        chunk_size = self.CHUNK_SIZE
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            for col in variables_numericas:
                if chunk[col].isnull().any():
                    chunk[col] = chunk[col].interpolate(method='time')
            df.iloc[i:i+chunk_size] = chunk
        
        # Llenar valores restantes
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    def categorizar_clima(self, row):
        """Categoriza el clima basado en múltiples variables"""
        categorias = []
        
        # Categorización por temperatura
        if row['temperatura_C'] < 8:
            categorias.append('Frío')
        elif row['temperatura_C'] < 16:
            categorias.append('Templado')
        else:
            categorias.append('Cálido')
            
        # Categorización por humedad
        if row['humedad_relativa'] > 85:
            categorias.append('Muy Húmedo')
        elif row['humedad_relativa'] > 70:
            categorias.append('Húmedo')
            
        # Categorización por precipitación
        if row['precipitacion_mm'] > 0.002:
            categorias.append('Lluvia Fuerte')
        elif row['precipitacion_mm'] > 0:
            categorias.append('Llovizna')
            
        # Categorización por nubosidad
        if row['cobertura_nubes_octas'] > 6:
            categorias.append('Muy Nublado')
        elif row['cobertura_nubes_octas'] > 4:
            categorias.append('Parcialmente Nublado')
            
        # Categorización por radiación solar
        if row['radiacion_solar_J_m2'] > 800:
            categorias.append('Alta Radiación')
        elif row['radiacion_solar_J_m2'] > 400:
            categorias.append('Radiación Moderada')
        
        return ' + '.join(categorias) if categorias else 'Normal'
    def preparar_categorias(self, df):
        """Prepara el conjunto completo de categorías antes del entrenamiento"""
        try:
            print("Preparando categorías globales...")
            todas_categorias = set()
            
            # Obtener todas las categorías posibles
            print("Analizando datos para encontrar todas las categorías posibles...")
            for _, row in df.iterrows():
                categoria = self.categorizar_clima(row)
                todas_categorias.add(categoria)
            
            # Ordenar y codificar todas las categorías
            self.categorias = sorted(list(todas_categorias))
            self.num_categorias = len(self.categorias)
            self.label_encoder.fit(self.categorias)  # Ajustar el encoder con todas las categorías
            
            print(f"\nTotal de categorías encontradas: {self.num_categorias}")
            print("\nCategorías detectadas:")
            for i, cat in enumerate(self.categorias, 1):
                print(f"{i}. {cat}")
                
            return self.num_categorias
            
        except Exception as e:
            print(f"Error en preparación de categorías: {str(e)}")
            raise Exception(f"Error en preparación de categorías: {str(e)}")
    def preparar_datos(self, df, ventana_tiempo=12):
        """Prepara los datos para el entrenamiento con procesamiento optimizado"""
        try:
            print("Iniciando preparación de datos...")
            
            # Crear copia del DataFrame para evitar warnings
            df = df.copy()
            
            # Generar categorías usando el encoder ya ajustado
            df['categoria_clima'] = df.apply(self.categorizar_clima, axis=1)
            df['categoria_numerica'] = self.label_encoder.transform(df['categoria_clima'])
            
            print(f"\nInformación de categorías:")
            print(f"Número total de categorías: {self.num_categorias}")
            print("Categorías encontradas:")
            for i, cat in enumerate(sorted(df['categoria_clima'].unique()), 1):
                print(f"{i}. {cat}")
            
            # Variables para normalizar
            variables_numericas = ['temperatura_C', 'humedad_relativa', 'precipitacion_mm',
                            'cobertura_nubes_octas', 'velocidad_viento_kmh', 
                            'radiacion_solar_J_m2']
            
            # Normalizar datos por chunks
            df_norm = pd.DataFrame()
            chunk_size = min(1000, len(df))  # Ajustar tamaño del chunk según memoria disponible
            
            for chunk_start in range(0, len(df), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(df))
                chunk = df.iloc[chunk_start:chunk_end].copy()
                
                for variable in variables_numericas:
                    # Usar scalers pre-entrenados si existen
                    if hasattr(self.scalers[variable], 'mean_'):
                        chunk[variable] = self.scalers[variable].transform(
                            chunk[variable].values.reshape(-1, 1)
                        )
                    else:
                        chunk[variable] = self.scalers[variable].fit_transform(
                            chunk[variable].values.reshape(-1, 1)
                        )
                df_norm = pd.concat([df_norm, chunk])

            print("\nProcesando secuencias de datos...")
            # Procesar datos en secuencias
            total_steps = len(df_norm) - ventana_tiempo - 72 + 1
            X_final = []
            y_final = []
            
            # Usar chunks más pequeños para las secuencias
            sequence_chunk_size = min(1000, total_steps)
            
            for start_idx in range(0, total_steps, sequence_chunk_size):
                end_idx = min(start_idx + sequence_chunk_size, total_steps)
                X_chunk = []
                y_chunk = []
                
                print(f"Procesando secuencias {start_idx} a {end_idx} de {total_steps}")
                
                for i in range(start_idx, end_idx):
                    try:
                        # Ventana de tiempo para entrada
                        ventana = df_norm[variables_numericas].iloc[i:(i+ventana_tiempo)].values
                        if len(ventana) != ventana_tiempo:
                            continue
                        X_chunk.append(ventana)
                        
                        # Próximas 72 horas para predicción
                        y_seq = df_norm['categoria_numerica'].iloc[i+ventana_tiempo:i+ventana_tiempo+72].values
                        if len(y_seq) != 72:
                            continue
                            
                        y_seq_onehot = np.zeros((72, self.num_categorias))
                        for t, cat in enumerate(y_seq):
                            if 0 <= cat < self.num_categorias:  # Validación adicional
                                y_seq_onehot[t, int(cat)] = 1
                        y_chunk.append(y_seq_onehot)
                        
                    except Exception as e:
                        print(f"Error procesando secuencia {i}: {str(e)}")
                        continue
                
                if X_chunk and y_chunk:  # Verificar que no estén vacíos
                    X_final.extend(X_chunk)
                    y_final.extend(y_chunk)
                
                # Liberar memoria explícitamente
                del X_chunk
                del y_chunk
                gc.collect()
            
            # Convertir a arrays numpy solo si hay datos
            if not X_final or not y_final:
                raise ValueError("No se pudieron generar secuencias válidas")
                
            X = np.array(X_final)
            y = np.array(y_final)
            
            # Verificar formas finales
            print(f"\nDimensiones finales:")
            print(f"X: {X.shape}")
            print(f"y: {y.shape}")
            print(f"Número de categorías en y: {y.shape[-1]}")
            
            # Verificar que las dimensiones sean correctas
            if y.shape[-1] != self.num_categorias:
                raise ValueError(f"Error de dimensiones: y tiene {y.shape[-1]} categorías pero deberían ser {self.num_categorias}")
                
            # Verificar que no haya valores NaN
            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError("Se encontraron valores NaN en los datos procesados")
            
            return X, y
            
        except Exception as e:
            print(f"Error en la preparación de datos: {str(e)}")
            raise Exception(f"Error en la preparación de datos: {str(e)}")
        finally:
            # Limpiar memoria
            if 'df_norm' in locals():
                del df_norm
            gc.collect()
    class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
        """Callback para ajuste gradual de learning rate"""
        def __init__(self, warmup_epochs=10, initial_lr=0.001):
            super().__init__()
            self.warmup_epochs = warmup_epochs
            self.initial_lr = initial_lr
            
        def on_epoch_begin(self, epoch, logs=None):
            if epoch < self.warmup_epochs:
                lr = (epoch + 1) * self.initial_lr / self.warmup_epochs
                self.model.optimizer._set_hyper('learning_rate', lr)

    def crear_modelo(self, input_shape, num_categorias):
        """Crea una arquitectura optimizada del modelo LSTM"""
        try:
            print(f"Creando modelo con input_shape: {input_shape} y {num_categorias} categorías")
            
            # Verificar dimensiones
            print(f"Número de categorías detectadas: {num_categorias}")
            
            model = keras.Sequential([
                # Capa de entrada con ruido gaussiano
                keras.layers.Input(shape=input_shape),
                keras.layers.GaussianNoise(0.1),
                
                # Primera capa LSTM
                keras.layers.LSTM(256, 
                    return_sequences=True,
                    kernel_regularizer=keras.regularizers.L2(0.01),
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                
                # Segunda capa LSTM
                keras.layers.LSTM(128,
                                kernel_regularizer=keras.regularizers.L2(0.01),
                                kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                
                # Capa densa intermedia
                keras.layers.Dense(256, 
                                activation='relu',
                                kernel_regularizer=keras.regularizers.L2(0.01),
                                kernel_initializer='he_normal'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.2),
                
                # Capa densa adicional para mejor capacidad de aprendizaje
                keras.layers.Dense(128, 
                                activation='relu',
                                kernel_regularizer=keras.regularizers.L2(0.01),
                                kernel_initializer='he_normal'),
                keras.layers.BatchNormalization(),
                
                # Capa de salida ajustada al número de categorías
                keras.layers.Dense(72 * num_categorias, 
                                activation='softmax',
                                kernel_initializer='glorot_uniform',
                                name='output_layer'),
                keras.layers.Reshape((72, num_categorias))
            ])
            
            # Verificar la forma de salida antes de compilar
            print(f"Forma de salida esperada: (None, 72, {num_categorias})")
            
            # Configurar optimizador con parámetros optimizados
            optimizer = keras.optimizers.Adam(
                learning_rate=self.LEARNING_RATE,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=True
            )
            
            # Compilar modelo con métricas y pérdida específicas
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    keras.metrics.AUC()
                ]
            )
            
            # Mostrar información detallada del modelo
            print("\nResumen del modelo:")
            model.summary()
            
            print("\nConfiguración del modelo:")
            print(f"Learning rate inicial: {self.LEARNING_RATE}")
            print(f"Regularización L2: 0.01")
            print(f"Dropout rates: 0.3, 0.3, 0.2")
            print(f"Forma de entrada: {input_shape}")
            print(f"Forma de salida: {model.output_shape}")
            
            return model
            
        except Exception as e:
            print(f"Error al crear el modelo: {str(e)}")
            raise Exception(f"Error en la creación del modelo: {str(e)}")

    def entrenar_modelo(self, df, epochs=200, batch_size=64, callback=None):
        """Entrena el modelo con manejo optimizado de memoria"""
        try:
            print("Configurando entrenamiento...")
            tf.keras.backend.clear_session()

            # Obtener todas las categorías posibles primero
            print("Analizando todas las categorías posibles...")
            todas_categorias = set()
            for _, row in df.iterrows():
                categoria = self.categorizar_clima(row)
                todas_categorias.add(categoria)
            
            self.categorias = sorted(list(todas_categorias))
            self.num_categorias = len(self.categorias)
            print(f"Total de categorías encontradas: {self.num_categorias}")
            # Ajustar el LabelEncoder con todas las categorías
            self.label_encoder.fit(self.categorias)
            # Configurar el modelo con el número correcto de categorías
            if self.model is None:
                X_sample, y_sample = self.preparar_datos(df[:1000], ventana_tiempo=12)
                input_shape = (X_sample.shape[1], X_sample.shape[2])
                self.model = self.crear_modelo(input_shape, self.num_categorias)

                # Configurar optimizador
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.LEARNING_RATE,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07
                )
                
                # Compilar modelo
                self.model.compile(
                    optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

            # Configurar callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=5,
                    restore_best_weights=True,
                    min_delta=0.001,
                    mode='min',
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'mejor_modelo_chunk_{epoch:02d}.keras',
                    save_best_only=True,
                    monitor='loss',
                    mode='min',
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001,
                    cooldown=2,
                    verbose=1
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1,
                    write_graph=True,
                    update_freq='epoch'
                )
            ]

            if callback:
                callbacks.append(
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_begin=lambda epoch, logs: print(f'\nIniciando época {epoch + 1}/{epochs}'),
                        on_epoch_end=lambda epoch, logs: callback(epoch, epochs)
                    )
                )
            
            history_list = []
            chunk_size = 10000
            total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
            
            print(f"\nIniciando entrenamiento por chunks ({total_chunks} chunks)...")
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(df))
                df_chunk = df[start_idx:end_idx].copy()
                
                print(f"\nProcesando chunk {chunk_idx + 1}/{total_chunks}")
                print(f"Registros {start_idx} a {end_idx}")
                
                try:
                    # Preparar datos del chunk
                    X_chunk, y_chunk = self.preparar_datos(df_chunk, ventana_tiempo=12)
                    
                    # Dividir en entrenamiento y validación
                    val_split = int(0.8 * len(X_chunk))
                    X_train = X_chunk[:val_split]
                    y_train = y_chunk[:val_split]
                    X_val = X_chunk[val_split:]
                    y_val = y_chunk[val_split:]
                    
                    # Crear datasets optimizados
                    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    train_dataset = train_dataset.shuffle(1000)
                    train_dataset = train_dataset.batch(batch_size)
                    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
                    
                    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                    val_dataset = val_dataset.batch(batch_size)
                    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
                    
                    # Entrenar en este chunk
                    print(f"\nEntrenando chunk {chunk_idx + 1}/{total_chunks}")
                    chunk_history = self.model.fit(
                        train_dataset,
                        validation_data=val_dataset,
                        epochs=5,
                        callbacks=callbacks,
                        verbose=1
                    )
                    history_list.append(chunk_history)
                    
                except Exception as e:
                    print(f"Error en chunk {chunk_idx + 1}: {str(e)}")
                    continue
                    
                # Limpiar memoria
                del X_chunk, y_chunk, X_train, y_train, X_val, y_val
                del train_dataset, val_dataset
                gc.collect()
                tf.keras.backend.clear_session()
                
            # Calcular y mostrar métricas finales
            if history_list:
                final_metrics = {
                    'loss': np.mean([h.history['loss'][-1] for h in history_list]),
                    'accuracy': np.mean([h.history['accuracy'][-1] for h in history_list]),
                    'val_loss': np.mean([h.history.get('val_loss', [0])[-1] for h in history_list]),
                    'val_accuracy': np.mean([h.history.get('val_accuracy', [0])[-1] for h in history_list])
                }
                
                print("\nMétricas finales del entrenamiento:")
                for metric, value in final_metrics.items():
                    print(f"{metric}: {value:.4f}")
                    
                # Guardar el mejor modelo
                try:
                    print("\nGuardando mejor modelo...")
                    self.model.save('mejor_modelo_final.keras')
                    print("Modelo guardado exitosamente")
                except Exception as e:
                    print(f"Error al guardar el modelo: {str(e)}")
                
                return history_list[-1]
            
            print("\nEntrenamiento completado")
            return None

        except Exception as e:
            print(f"Error en el entrenamiento: {str(e)}")
            return None
        finally:
            # Limpiar archivos temporales
            for file in glob.glob('mejor_modelo_chunk_*.keras'):
                try:
                    os.remove(file)
                except:
                    pass
    def _guardar_modelo_y_metricas(self, history):
        """Método auxiliar para guardar modelo y métricas"""
        try:
            print("\nGuardando modelo final...")
            # Asegurarse de que el directorio existe
            modelo_dir = 'modelos'
            if not os.path.exists(modelo_dir):
                os.makedirs(modelo_dir)
                
            # Guardar con timestamp para evitar sobreescrituras
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            modelo_path = os.path.join(modelo_dir, f'modelo_microclima_{timestamp}.keras')
            self.model.save(modelo_path)
            print(f"Modelo guardado exitosamente en: {modelo_path}")
            
            # Guardar métricas
            print("\nMétricas finales del entrenamiento:")
            metricas_finales = {
                'accuracy': history.history['accuracy'][-1],
                'loss': history.history['loss'][-1],
                'val_accuracy': history.history['val_accuracy'][-1],
                'val_loss': history.history['val_loss'][-1]
            }
            
            for metrica, valor in metricas_finales.items():
                print(f"{metrica}: {valor:.4f}")
            
            # Guardar métricas en archivo
            metricas_path = os.path.join(modelo_dir, f'metricas_{timestamp}.npy')
            np.save(metricas_path, history.history)
            print(f"Métricas guardadas exitosamente en: {metricas_path}")
            
        except Exception as e:
            print(f"Error al guardar el modelo o métricas: {str(e)}")
            raise Exception(f"Error al guardar el modelo o métricas: {str(e)}")

    def cargar_modelo_guardado(self):
        """Carga un modelo previamente guardado"""
        try:
            if os.path.exists(self.modelo_path):
                print("Cargando modelo guardado...")
                self.model = tf.keras.models.load_model(self.modelo_path)
                print("Modelo cargado exitosamente")
                return True
            else:
                raise Exception("No se encontró el archivo del modelo")
        except Exception as e:
            raise Exception(f"Error al cargar el modelo: {str(e)}")
    def predecir_temperatura(self, datos_entrada, hora_futura):
        """Predice la temperatura con ajustes optimizados"""
        try:
            # Obtener temperatura actual
            temp_actual = self.scalers['temperatura_C'].inverse_transform(
                datos_entrada[0, -1:, 0].reshape(-1, 1)
            )[0][0]
            
            # Calcular hora del día
            hora = (datetime.now() + timedelta(hours=hora_futura)).hour
            
            # Factores de variación según hora
            if 6 <= hora < 12:  # Mañana
                factor = 0.15
                variacion = 0.8
            elif 12 <= hora < 18:  # Tarde
                factor = 0.25
                variacion = 1.0
            elif 18 <= hora < 22:  # Noche
                factor = -0.15
                variacion = 0.6
            else:  # Madrugada
                factor = -0.25
                variacion = 0.4
                
            # Predicción con variación controlada
            temperatura_predicha = temp_actual + (factor * np.random.normal(0, variacion))
            return max(min(temperatura_predicha, 35), 0)  # Limitar entre 0°C y 35°C
                
        except Exception as e:
            print(f"Error en predicción de temperatura: {str(e)}")
            return 20.0  # Valor por defecto

    def predecir_proximo_periodo(self, dataset):
        """Genera predicciones optimizadas para las próximas 72 horas"""
        try:
            if self.model is None:
                raise Exception("El modelo no está entrenado o cargado")

            # Preparar datos para predicción
            X, _ = self.preparar_datos(dataset)
            ultimos_datos = X[-1:]
            
            # Realizar predicción con softmax para obtener probabilidades reales
            predicciones_raw = self.model.predict(ultimos_datos, batch_size=1)
            predicciones_prob = predicciones_raw[0]
            
            resultados = []
            fechas_prediccion = pd.date_range(
                start=dataset.index[-1] + pd.Timedelta(hours=1),
                periods=72,
                freq='h'
            )
            
                    # Procesar predicciones
            for i, probs in enumerate(predicciones_prob):
                categoria_idx = np.argmax(probs)
                # Obtener la probabilidad real usando softmax
                confianza = float(tf.nn.softmax(probs)[categoria_idx])
                categoria = self.label_encoder.inverse_transform([categoria_idx])[0]
                
                temperatura = self.predecir_temperatura(ultimos_datos, i)
                
                prediccion = {
                    'fecha': fechas_prediccion[i].strftime('%Y-%m-%d %H:%M'),
                    'hora': fechas_prediccion[i].strftime('%H:%M'),
                    'categoria': categoria,
                    'confianza': confianza * 100,  # Convertir a porcentaje
                    'temperatura': temperatura,
                    'detalles': self.generar_detalles_prediccion(categoria, confianza)
                }
                
                resultados.append(prediccion)
            
            return resultados
                
        except Exception as e:
            raise Exception(f"Error en predicción: {str(e)}")

    def generar_detalles_prediccion(self, categoria, confianza):
        """Genera detalles enriquecidos para la predicción"""
        return {
            'descripcion': self.generar_descripcion_clima(categoria),
            'nivel_confianza': self.clasificar_confianza(confianza),
            'recomendaciones': self.generar_recomendaciones(categoria),
            'indice_confiabilidad': round(confianza * 100, 2)
        }

    def generar_descripcion_clima(self, categoria):
        """Genera descripción detallada del clima predicho"""
        partes = categoria.split(' + ')
        descripcion = "Se espera un clima "
        return descripcion + " con ".join(parte.lower() for parte in partes)

    def clasificar_confianza(self, confianza):
        """Clasifica el nivel de confianza"""
        if confianza > 0.85:
            return "Muy Alta"
        elif confianza > 0.70:
            return "Alta"
        elif confianza > 0.50:
            return "Moderada"
        return "Baja"

    def generar_recomendaciones(self, categoria):
        """Genera recomendaciones basadas en la categoría"""
        recomendaciones = []
        
        if "Lluvia Fuerte" in categoria:
            recomendaciones.extend([
                "Llevar paraguas o impermeable",
                "Evitar zonas propensas a inundaciones"
            ])
        if "Llovizna" in categoria:
            recomendaciones.append("Considerar llevar protección para la lluvia")
        if "Frío" in categoria:
            recomendaciones.extend([
                "Abrigarse bien",
                "Evitar exposición prolongada al frío"
            ])
        if "Cálido" in categoria:
            recomendaciones.extend([
                "Mantenerse hidratado",
                "Usar protección solar",
                "Evitar actividades al aire libre en horas pico"
            ])
        if "Muy Nublado" in categoria:
            recomendaciones.extend([
                "Precaución al conducir",
                "Mantener las luces encendidas"
            ])
                
        return recomendaciones