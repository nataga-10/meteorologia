import pandas as pd
import os
from datetime import datetime
import glob
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tkcalendar import DateEntry  # Necesitarás instalar este paquete: pip install tkcalendar

def preparar_datos_estacion(ruta_original, ruta_salida=None, convertir_a_horario=True, verbose=True, rellenar_faltantes=True):
    """
    Prepara los datos de la estación meteorológica para su uso en el modelo de predicción:
    1. Convierte radiación solar de W/m² a J/m²
    2. Opcionalmente convierte datos cada 5 minutos a datos horarios usando el valor más cercano a cada hora
    3. Rellena valores faltantes de radiación solar con el último valor observado
    """
    # Generar nombre de archivo de salida si no se proporciona
    if ruta_salida is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directorio = os.path.dirname(ruta_original)
        if convertir_a_horario:
            ruta_salida = os.path.join(directorio, f"datos_horarios_{timestamp}.csv")
        else:
            ruta_salida = os.path.join(directorio, f"datos_convertidos_{timestamp}.csv")
    
    print(f"Procesando archivo: {ruta_original}")
    
    # Intentar detectar el delimitador del archivo
    with open(ruta_original, 'r') as f:
        primera_linea = f.readline().strip()
    
    # Determinar el delimitador
    if ';' in primera_linea:
        delimitador = ';'
        print("Detectado delimitador: punto y coma (;)")
    elif '\t' in primera_linea:
        delimitador = '\t'
        print("Detectado delimitador: tabulador (\\t)")
    else:
        delimitador = ','
        print("Detectado delimitador: coma (,)")
    
    # Cargar dataset original con el delimitador detectado
    df = pd.read_csv(ruta_original, delimiter=delimitador)
    print(f"Registros originales: {len(df)}")
    
    # Mostrar columnas disponibles
    print(f"Columnas disponibles: {df.columns.tolist()}")
    
    # Renombrar la columna de fecha si es necesario
    if 'fecha_hora' in df.columns and 'fecha' not in df.columns:
        df = df.rename(columns={'fecha_hora': 'fecha'})
        print("Columna 'fecha_hora' renombrada a 'fecha'")
    
    # Verificar y seleccionar columnas necesarias
    columnas_necesarias = ['fecha']
    mapeo_columnas = {
        'temp_dht_raw': 'temperatura_C',
        'hum_dht_raw': 'humedad_relativa',
        'lluvia_mm': 'precipitacion_mm',
        'cobertura_nubes_octas': 'cobertura_nubes_octas',
        'vel_viento_kmh': 'velocidad_viento_kmh',
        'radiacion_solar_wm2': 'radiacion_solar_wm2'
    }
    
    # Crear un nuevo DataFrame con la fecha
    nuevo_df = pd.DataFrame()
    if 'fecha' not in df.columns:
        print("ADVERTENCIA: No se encontró columna 'fecha'. Buscando alternativas...")
        # Buscar columnas que puedan contener la fecha
        posibles_columnas_fecha = [col for col in df.columns if 'fecha' in col.lower()]
        if posibles_columnas_fecha:
            print(f"Usando columna '{posibles_columnas_fecha[0]}' como fecha")
            nuevo_df['fecha'] = df[posibles_columnas_fecha[0]]
        else:
            # Si no hay columna con 'fecha' en el nombre, intentar usar la primera columna
            print("Usando primera columna como fecha")
            nuevo_df['fecha'] = df.iloc[:, 0]
    else:
        nuevo_df['fecha'] = df['fecha']
    
    # Añadir columnas necesarias con mapeo
    for col_orig, col_nueva in mapeo_columnas.items():
        # Buscar la columna original o variantes
        columna_encontrada = None
        if col_orig in df.columns:
            columna_encontrada = col_orig
        else:
            # Buscar variantes (ej: si contiene el nombre)
            posibles_columnas = [col for col in df.columns if col_orig in col.lower()]
            if posibles_columnas:
                columna_encontrada = posibles_columnas[0]
            else:
                # Buscar por coincidencia parcial
                posibles_columnas = [col for col in df.columns if col_nueva.lower() in col.lower()]
                if posibles_columnas:
                    columna_encontrada = posibles_columnas[0]
        
        if columna_encontrada:
            nuevo_df[col_nueva] = df[columna_encontrada]
            print(f"Columna '{columna_encontrada}' mapeada a '{col_nueva}'")
        else:
            print(f"Advertencia: Columna '{col_orig}' no encontrada")
    
    # Verificar si la columna de radiación solar tiene otro nombre
    if 'radiacion_solar_wm2' not in nuevo_df.columns:
        radiacion_cols = [col for col in df.columns if 'radiacion' in col.lower() or 'solar' in col.lower()]
        if radiacion_cols:
            col_rad = radiacion_cols[0]
            nuevo_df['radiacion_solar_wm2'] = df[col_rad]
            print(f"Usando columna '{col_rad}' como radiación solar")
    
    # Si hay una columna radiacion_solar_J_m2 pero no radiacion_solar_wm2, usar la primera
    if 'radiacion_solar_wm2' not in nuevo_df.columns and 'radiacion_solar_J_m2' in df.columns:
        nuevo_df['radiacion_solar_J_m2'] = df['radiacion_solar_J_m2']
        print("Usando directamente la columna 'radiacion_solar_J_m2'")
    
    # Limpiar posibles espacios en blanco en los datos
    for col in nuevo_df.columns:
        if nuevo_df[col].dtype == object:  # Solo para columnas de texto
            nuevo_df[col] = nuevo_df[col].str.strip() if hasattr(nuevo_df[col], 'str') else nuevo_df[col]
    
    # Convertir todas las columnas numéricas a valores numéricos
    for col in nuevo_df.columns:
        if col != 'fecha':
            nuevo_df[col] = pd.to_numeric(nuevo_df[col], errors='coerce')
    
    # Convertir radiación solar si es necesario
    if 'radiacion_solar_wm2' in nuevo_df.columns and 'radiacion_solar_J_m2' not in nuevo_df.columns:
        print("\nConvirtiendo radiación solar de W/m² a J/m²...")
        
        # Mostrar primeros valores
        print("Algunos valores de radiacion_solar_wm2:")
        print(nuevo_df['radiacion_solar_wm2'].head(3).values)
        
        # Convertir de W/m² a J/m² (5 minutos = 300 segundos)
        periodo_segundos = 300
        nuevo_df['radiacion_solar_J_m2'] = nuevo_df['radiacion_solar_wm2'] * periodo_segundos
        
        # Mostrar primeros valores convertidos
        print("Valores convertidos a J/m²:")
        print(nuevo_df['radiacion_solar_J_m2'].head(3).values)
        
        # Eliminar columna original
        nuevo_df = nuevo_df.drop('radiacion_solar_wm2', axis=1)
        print("Columna 'radiacion_solar_wm2' eliminada tras la conversión")
    else:
        print("No se realizó conversión de radiación solar (Ya está en J/m² o no se encontró la columna)")
    
    # Verificar y rellenar valores faltantes en radiación solar
    if 'radiacion_solar_J_m2' in nuevo_df.columns and rellenar_faltantes:
        # 1. Asegurar que toda la columna es numérica (convierte cadenas vacías a NaN)
        nuevo_df['radiacion_solar_J_m2'] = pd.to_numeric(nuevo_df['radiacion_solar_J_m2'], errors='coerce')
        
        # 2. Contar valores nulos antes del relleno
        nulos_antes = nuevo_df['radiacion_solar_J_m2'].isna().sum()
        if nulos_antes > 0:
            print(f"\nDetectados {nulos_antes} valores faltantes en radiación solar")
            
            # 3. Rellenar usando el método "último valor observado"
            ultimo_valor = None
            for idx in nuevo_df.index:
                if pd.isna(nuevo_df.at[idx, 'radiacion_solar_J_m2']):
                    if ultimo_valor is not None:
                        nuevo_df.at[idx, 'radiacion_solar_J_m2'] = ultimo_valor
                        if verbose:
                            print(f"Rellenado valor en índice {idx} (fecha: {nuevo_df.at[idx, 'fecha']}) con {ultimo_valor}")
                else:
                    ultimo_valor = nuevo_df.at[idx, 'radiacion_solar_J_m2']
            
            # 4. Comprobar si quedaron valores nulos (por si el primer valor era nulo)
            nulos_despues = nuevo_df['radiacion_solar_J_m2'].isna().sum()
            if nulos_despues > 0 and nulos_despues < nulos_antes:
                # Si quedan valores nulos al inicio, usar el primer valor no nulo
                primer_valor_valido = nuevo_df.loc[~nuevo_df['radiacion_solar_J_m2'].isna(), 'radiacion_solar_J_m2'].iloc[0]
                nuevo_df['radiacion_solar_J_m2'] = nuevo_df['radiacion_solar_J_m2'].fillna(primer_valor_valido)
                if verbose:
                    print(f"Rellenado {nulos_despues} valores iniciales con {primer_valor_valido}")
                nulos_despues = 0
            
            print(f"Valores faltantes rellenados: {nulos_antes - nulos_despues}")
            print(f"Valores faltantes restantes: {nulos_despues}")
    
    # Corregir valores extremos
    print("\nVerificando y corrigiendo valores extremos...")
    
    if 'temperatura_C' in nuevo_df.columns:
        val_invalidos = ((nuevo_df['temperatura_C'] < -10) | (nuevo_df['temperatura_C'] > 35)).sum()
        if val_invalidos > 0:
            nuevo_df.loc[nuevo_df['temperatura_C'] < -10, 'temperatura_C'] = None
            nuevo_df.loc[nuevo_df['temperatura_C'] > 35, 'temperatura_C'] = None
            print(f"Corregidos {val_invalidos} valores extremos de temperatura")
    
    if 'humedad_relativa' in nuevo_df.columns:
        val_invalidos = ((nuevo_df['humedad_relativa'] < 0) | (nuevo_df['humedad_relativa'] > 100)).sum()
        if val_invalidos > 0:
            nuevo_df.loc[nuevo_df['humedad_relativa'] < 0, 'humedad_relativa'] = None
            nuevo_df.loc[nuevo_df['humedad_relativa'] > 100, 'humedad_relativa'] = None
            print(f"Corregidos {val_invalidos} valores extremos de humedad")
    
    # Convertir fecha a datetime
    try:
        nuevo_df['fecha'] = pd.to_datetime(nuevo_df['fecha'])
    except:
        print("Error al convertir la columna de fecha. Intentando diferentes formatos...")
        try:
            # Intentar diferentes formatos
            nuevo_df['fecha'] = pd.to_datetime(nuevo_df['fecha'], format='%d/%m/%Y %H:%M', errors='coerce')
        except:
            try:
                nuevo_df['fecha'] = pd.to_datetime(nuevo_df['fecha'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            except:
                try:
                    # Formato d/mm/yyyy
                    nuevo_df['fecha'] = pd.to_datetime(nuevo_df['fecha'], format='%d/%m/%Y %H:%M', errors='coerce')
                except:
                    print("No se pudo convertir la columna de fecha a formato datetime.")
                    return None
    
    # Eliminar filas con fechas inválidas
    filas_antes = len(nuevo_df)
    nuevo_df = nuevo_df.dropna(subset=['fecha'])
    filas_despues = len(nuevo_df)
    if filas_antes > filas_despues:
        print(f"Se eliminaron {filas_antes - filas_despues} filas con fechas inválidas")
    
    # Si se requiere resampling a formato horario
    if convertir_a_horario:
        print("\nConvirtiendo datos a formato horario...")
        
        # Primero asegurarse de que el DataFrame esté ordenado por fecha
        nuevo_df = nuevo_df.sort_values('fecha')
        
        # Crear un DataFrame vacío para los resultados
        df_procesado = pd.DataFrame(columns=nuevo_df.columns)
        
        # Obtener la fecha mínima y máxima
        fecha_min = nuevo_df['fecha'].min().floor('D')
        fecha_max = nuevo_df['fecha'].max().ceil('D')
        
        # Crear rango de horas
        horas_exactas = pd.date_range(start=fecha_min, end=fecha_max, freq='1H')
        
        # Para cada hora exacta, buscar el registro más cercano
        for hora_exacta in horas_exactas:
            # Obtener solo registros del mismo día
            mismo_dia = nuevo_df['fecha'].dt.date == hora_exacta.date()
            
            # Calcular la diferencia en minutos entre cada registro y la hora exacta
            diferencias = abs((nuevo_df.loc[mismo_dia, 'fecha'] - hora_exacta).dt.total_seconds() / 60)
            
            if not diferencias.empty:
                # Encontrar el índice del registro más cercano
                idx_cercano = diferencias.idxmin()
                registro_cercano = nuevo_df.loc[idx_cercano].copy()
                
                # Para comprobar: mostrar la fecha original y la diferencia en minutos
                fecha_original = registro_cercano['fecha']
                diferencia_min = diferencias.min() / 60  # Convertir a horas para más claridad
                
                if verbose and diferencia_min > 0.25:  # Si la diferencia es mayor a 15 minutos
                    print(f"ADVERTENCIA: Para {hora_exacta}, registro más cercano: {fecha_original}, diferencia: {diferencia_min:.2f} horas")
                
                # Modificar la fecha al formato de hora exacta
                registro_cercano['fecha'] = hora_exacta
                
                # Agregar al DataFrame de resultados
                df_procesado = pd.concat([df_procesado, pd.DataFrame([registro_cercano])], ignore_index=True)
        
        # Para radiación solar, si se desea un promedio en lugar del valor más cercano
        if 'radiacion_solar_J_m2' in nuevo_df.columns:
            print("Calculando promedio de radiación solar para cada hora...")
            
            # Establecer el índice temporal para el cálculo del promedio de radiación
            temp_df = nuevo_df.set_index('fecha')
            
            # Calcular el promedio horario solo para radiación solar
            radiacion_promedio = temp_df['radiacion_solar_J_m2'].resample('1H').mean()
            
            # Reemplazar los valores de radiación solar con los promedios
            for idx, hora in enumerate(df_procesado['fecha']):
                if hora in radiacion_promedio.index:
                    df_procesado.loc[idx, 'radiacion_solar_J_m2'] = radiacion_promedio[hora]
        
        print(f"Datos convertidos a formato horario: {len(df_procesado)} registros")
    else:
        df_procesado = nuevo_df
        print(f"Se mantiene la frecuencia original: {len(df_procesado)} registros")
    
    # Guardar el resultado
    df_procesado.to_csv(ruta_salida, index=False)
    
    print(f"\nDatos procesados guardados en: {ruta_salida}")
    print(f"Rango de fechas: {df_procesado['fecha'].min()} a {df_procesado['fecha'].max()}")
    
    # Como verificación adicional, mostrar algunos ejemplos de conversión
    if convertir_a_horario and verbose:
        print("\nEjemplos de conversión (original → procesado):")
        # Seleccionar algunas horas al azar para verificar
        for hora in df_procesado['fecha'].sample(min(5, len(df_procesado))):
            hora_redondeada = hora.strftime('%Y-%m-%d %H:00')
            
            # Encontrar el registro original más cercano a esta hora
            diferencias = abs((nuevo_df['fecha'] - hora).dt.total_seconds())
            idx_cercano = diferencias.idxmin()
            registro_orig = nuevo_df.loc[idx_cercano]
            
            # Encontrar el registro procesado para esta hora
            registro_proc = df_procesado[df_procesado['fecha'] == hora]
            
            if not registro_proc.empty:
                print(f"Hora: {hora_redondeada}")
                cols_to_show = [col for col in df_procesado.columns if col != 'fecha' and col in registro_orig.index]
                for col in cols_to_show[:3]:  # Mostrar solo las primeras 3 columnas para brevedad
                    print(f"  {col}: Original={registro_orig[col]}, Procesado={registro_proc[col].values[0]}")
    
    return ruta_salida

class EstacionMeteorologicaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Datos Meteorológicos")
        self.root.geometry("800x600")
        self.root.minsize(600, 500)  # Tamaño mínimo para asegurar que los elementos sean visibles
        self.root.resizable(True, True)
        
        # Configurar el tema y colores de fondo
        self.style = ttk.Style()
        self.style.configure("TFrame", background="white")
        self.style.configure("TLabelframe", background="white")
        self.style.configure("TLabelframe.Label", background="white")
        self.style.configure("TButton", background="white")
        self.style.configure("TCheckbutton", background="white")
        self.style.configure("TLabel", background="white")
        
        # Establecer ícono si está disponible
        try:
            self.root.iconbitmap("icono.ico")  # Reemplaza con la ruta a tu ícono si lo tienes
        except:
            pass
        
        # Variables para almacenar la configuración
        self.ruta_archivo = tk.StringVar()
        self.convertir_horario = tk.BooleanVar(value=True)
        self.fecha_inicio = tk.StringVar()
        self.fecha_fin = tk.StringVar()
        self.ruta_salida = tk.StringVar()
        self.filtrar_fechas = tk.BooleanVar(value=False)
        self.rellenar_faltantes = tk.BooleanVar(value=True)
        
        # Configurar color de fondo para la ventana principal
        self.root.configure(background="white")
        
        # Crear el marco principal
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar que los pesos de las filas y columnas funcionen correctamente
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(4, weight=1)  # La fila del log debe expandirse
        
        # Sección de selección de archivo
        archivo_frame = ttk.LabelFrame(self.main_frame, text="Selección de Archivo", padding="10")
        archivo_frame.pack(fill=tk.X, pady=10)
        
        archivo_frame.columnconfigure(1, weight=1)  # La columna del entry debe expandirse
        
        ttk.Label(archivo_frame, text="Archivo CSV:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(archivo_frame, textvariable=self.ruta_archivo, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(archivo_frame, text="Examinar...", command=self.buscar_archivo).grid(row=0, column=2, padx=5, pady=5)
        
        # Sección de opciones de procesamiento
        opciones_frame = ttk.LabelFrame(self.main_frame, text="Opciones de Procesamiento", padding="10")
        opciones_frame.pack(fill=tk.X, pady=10)
        
        ttk.Checkbutton(opciones_frame, text="Convertir a formato horario", variable=self.convertir_horario).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(opciones_frame, text="Rellenar valores faltantes con último valor observado", variable=self.rellenar_faltantes).grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # Sección de filtro de fechas
        filtro_frame = ttk.LabelFrame(self.main_frame, text="Filtro de Fechas", padding="10")
        filtro_frame.pack(fill=tk.X, pady=10)
        
        ttk.Checkbutton(filtro_frame, text="Aplicar filtro de fechas", variable=self.filtrar_fechas).grid(row=0, column=0, sticky=tk.W, pady=5, columnspan=2)
        
        ttk.Label(filtro_frame, text="Fecha de inicio:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.fecha_inicio_picker = DateEntry(filtro_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
        self.fecha_inicio_picker.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(filtro_frame, text="Fecha de fin:").grid(row=1, column=2, sticky=tk.W, pady=5)
        self.fecha_fin_picker = DateEntry(filtro_frame, width=12, background='darkblue', foreground='white', borderwidth=2)
        self.fecha_fin_picker.grid(row=1, column=3, padx=5, pady=5)
        
        # Sección de destino
        destino_frame = ttk.LabelFrame(self.main_frame, text="Archivo de Salida", padding="10")
        destino_frame.pack(fill=tk.X, pady=10)
        
        destino_frame.columnconfigure(1, weight=1)  # La columna del entry debe expandirse
        
        ttk.Label(destino_frame, text="Guardar en:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(destino_frame, textvariable=self.ruta_salida, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(destino_frame, text="Examinar...", command=self.seleccionar_destino).grid(row=0, column=2, padx=5, pady=5)
        
        # Barra de progreso (inicialmente oculta)
        progreso_frame = ttk.Frame(self.main_frame)
        progreso_frame.pack(fill=tk.X, pady=5)
        
        self.barra_progreso = ttk.Progressbar(progreso_frame, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
        self.barra_progreso.pack(fill=tk.X, padx=5)
        self.barra_progreso.pack_forget()  # Ocultar hasta que se necesite
        
        # Log de procesamiento con fondo oscuro y texto verde
        log_frame = ttk.LabelFrame(self.main_frame, text="Registro de Operaciones", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Configurar el log con fondo oscuro y texto verde
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, width=80, height=10,
                                bg="#1E1E1E", fg="#00FF00", insertbackground="#00FF00",
                                font=("Consolas", 10))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Botones de acción en un frame que se expande horizontalmente
        botones_frame = ttk.Frame(self.main_frame)
        botones_frame.pack(fill=tk.X, pady=10)
        
        # Colocar los botones a la derecha pero con suficiente espacio
        ttk.Button(botones_frame, text="Procesar", command=self.procesar_datos).pack(side=tk.RIGHT, padx=5)
        ttk.Button(botones_frame, text="Limpiar", command=self.limpiar_campos).pack(side=tk.RIGHT, padx=5)
        
        # Inicializar log
        self.log("Aplicación iniciada. Por favor, seleccione un archivo CSV para procesar.")
    
    def log(self, mensaje):
        """Añade un mensaje al registro de operaciones con marca de tiempo"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {mensaje}\n")
        self.log_text.see(tk.END)  # Desplazar al final
        self.root.update_idletasks()  # Actualizar la interfaz para mostrar el mensaje inmediatamente
    
    def buscar_archivo(self):
        """Abre un diálogo para seleccionar un archivo CSV"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if filepath:
            self.ruta_archivo.set(filepath)
            self.log(f"Archivo seleccionado: {filepath}")
            
            # Sugerir una ruta de salida automáticamente
            directorio = os.path.dirname(filepath)
            nombre_base = os.path.basename(filepath).split('.')[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.convertir_horario.get():
                sugerencia = os.path.join(directorio, f"{nombre_base}_horario_{timestamp}.csv")
            else:
                sugerencia = os.path.join(directorio, f"{nombre_base}_procesado_{timestamp}.csv")
            
            self.ruta_salida.set(sugerencia)
    
    def seleccionar_destino(self):
        """Abre un diálogo para seleccionar dónde guardar el archivo procesado"""
        filepath = filedialog.asksaveasfilename(
            title="Guardar archivo procesado como",
            defaultextension=".csv",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if filepath:
            self.ruta_salida.set(filepath)
            self.log(f"Archivo de salida: {filepath}")
    
    def limpiar_campos(self):
        """Limpia todos los campos de entrada"""
        self.ruta_archivo.set("")
        self.ruta_salida.set("")
        self.log("Campos limpiados.")
    
    def iniciar_progreso(self):
        """Muestra y activa la barra de progreso"""
        self.barra_progreso.pack(fill=tk.X, padx=5, pady=5)
        self.barra_progreso.start(10)  # Inicia la animación
        self.root.update_idletasks()
    
    def detener_progreso(self):
        """Detiene y oculta la barra de progreso"""
        self.barra_progreso.stop()
        self.barra_progreso.pack_forget()
        self.root.update_idletasks()
    
    def procesar_datos(self):
        """Procesa el archivo seleccionado con las opciones especificadas"""
        ruta_archivo = self.ruta_archivo.get()
        
        if not ruta_archivo:
            messagebox.showerror("Error", "Por favor, seleccione un archivo CSV para procesar.")
            return
            
        if not os.path.exists(ruta_archivo):
            messagebox.showerror("Error", f"El archivo {ruta_archivo} no existe.")
            return
        
        # Mostrar la barra de progreso
        self.iniciar_progreso()
        
        # Redireccionar la salida estándar al widget de log
        import sys
        original_stdout = sys.stdout
        
        class StdoutRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            
            def write(self, string):
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
                self.text_widget.update_idletasks()  # Actualizar la vista inmediatamente
            
            def flush(self):
                pass
        
        sys.stdout = StdoutRedirector(self.log_text)
        
        try:
            self.log("\n" + "="*60)
            self.log("INICIANDO PROCESAMIENTO DE ARCHIVO")
            self.log("="*60)
            
            # Obtener opciones
            convertir_horario = self.convertir_horario.get()
            ruta_salida = self.ruta_salida.get() if self.ruta_salida.get() else None
            rellenar_faltantes = self.rellenar_faltantes.get()
            
            # Procesar el archivo
            archivo_procesado = preparar_datos_estacion(
                ruta_archivo, 
                ruta_salida=ruta_salida,
                convertir_a_horario=convertir_horario,
                verbose=True,
                rellenar_faltantes=rellenar_faltantes
            )
            
            if archivo_procesado is None:
                raise Exception("Error en el procesamiento del archivo")
                
            # Aplicar filtro de fechas si está activado
            if self.filtrar_fechas.get():
                fecha_inicio = self.fecha_inicio_picker.get_date()
                fecha_fin = self.fecha_fin_picker.get_date()
                
                if fecha_inicio and fecha_fin:
                    self.log(f"\nAplicando filtro de fechas: {fecha_inicio} a {fecha_fin}")
                    
                    # Cargar el archivo procesado
                    df = pd.read_csv(archivo_procesado)
                    df['fecha'] = pd.to_datetime(df['fecha'])
                    
                    # Filtrar por fechas
                    df_filtrado = df[(df['fecha'].dt.date >= fecha_inicio) & 
                                    (df['fecha'].dt.date <= fecha_fin)]
                    
                    # Guardar archivo filtrado
                    nombre_filtrado = archivo_procesado.replace('.csv', f'_filtrado_{fecha_inicio.strftime("%Y%m%d")}_a_{fecha_fin.strftime("%Y%m%d")}.csv')
                    df_filtrado.to_csv(nombre_filtrado, index=False)
                    
                    self.log(f"Datos filtrados guardados en: {nombre_filtrado}")
                    self.log(f"Registros originales: {len(df)}, Registros filtrados: {len(df_filtrado)}")
                    
                    # Mostrar mensaje con resumen
                    messagebox.showinfo("Proceso Completado", 
                                       f"Procesamiento y filtrado completados.\n\n"
                                       f"Archivo original: {len(df)} registros\n"
                                       f"Archivo filtrado: {len(df_filtrado)} registros\n\n"
                                       f"Los datos han sido guardados en:\n{nombre_filtrado}")
                else:
                    self.log("Filtro de fechas seleccionado pero no se proporcionaron fechas válidas")
                    messagebox.showinfo("Proceso Completado", 
                                       "Procesamiento completado sin filtrar.\n\n"
                                       f"Los datos han sido guardados en:\n{archivo_procesado}")
            else:
                self.log("\n¡Procesamiento completado con éxito!")
                messagebox.showinfo("Proceso Completado", 
                                   f"Procesamiento completado.\n\n"
                                   f"Los datos han sido guardados en:\n{archivo_procesado}")
            
        except Exception as e:
            import traceback
            self.log(f"\nError al procesar el archivo: {str(e)}")
            traceback.print_exc(file=sys.stdout)
            messagebox.showerror("Error de Procesamiento", str(e))
        
        # Restaurar stdout
        sys.stdout = original_stdout
        
        # Detener y ocultar la barra de progreso
        self.detener_progreso()

if __name__ == "__main__":
    try:
        # Configurar tema para mejorar la apariencia
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")  # Puedes probar otros temas: "arc", "equilux", "breeze"
        
        # Configurar color de fondo blanco para ventana principal si usamos ttkthemes
        style = ttk.Style(root)
        style.configure(".", background="white")
        style.configure("TFrame", background="white")
        style.configure("TLabelframe", background="white")
        style.configure("TLabelframe.Label", background="white")
        root.configure(background="white")
        
    except ImportError:
        root = tk.Tk()
        root.configure(background="white")
        style = ttk.Style()
        style.theme_use('default')  # Usar el tema predeterminado
        
        # Configurar colores de fondo blancos para todos los widgets
        style.configure(".", background="white")
        style.configure("TFrame", background="white")
        style.configure("TLabelframe", background="white") 
        style.configure("TLabelframe.Label", background="white")
        
        root.tk.call('tk', 'scaling', 1.25)  # Escalar la interfaz para mejor visualización
    
    app = EstacionMeteorologicaGUI(root)
    root.mainloop()
######## By: Bryan Rojas and Nathalia Gutierrez ########
# 2024-01-01