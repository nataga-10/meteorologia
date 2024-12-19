import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import threading
from datetime import datetime
from predictor_model import PrediccionMicroclima
from visualizaciones import VisualizacionMicroclima
import seaborn as sns
import gc
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk



# Configuración para reducir los mensajes de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class VentanaProgreso(tk.Toplevel):
    def __init__(self, parent, titulo="Procesando"):
        super().__init__(parent)
        self.title(titulo)
        self.geometry("300x150")
        self.resizable(False, False)
        
        # Hacer la ventana modal
        self.transient(parent)
        self.grab_set()
        
        # Centrar la ventana
        self.center_window()
        
        # Frame principal
        self.frame = ttk.Frame(self)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Variables
        self.progress_var = tk.DoubleVar()
        
        # Widgets
        self.label = ttk.Label(self.frame, text="0%", font=('Arial', 10, 'bold'))
        self.label.pack(pady=10)
        
        self.progress = ttk.Progressbar(
            self.frame, 
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=200
        )
        self.progress.pack(pady=10)
        
        self.message = ttk.Label(self.frame, text="", font=('Arial', 9))
        self.message.pack(pady=10)
        
        # Protocolo de cierre
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def center_window(self):
        """Centra la ventana en la pantalla"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    def on_closing(self):
        """Maneja el intento de cerrar la ventana"""
        # Simplemente ignora el intento de cerrar
        pass
    
    def update_progress(self, value, message=""):
        """Actualiza el progreso de forma segura"""
        try:
            if self.winfo_exists():
                self.progress_var.set(value)
                self.label.config(text=f"{value:.1f}%")
                if message:
                    self.message.config(text=message)
                self.update_idletasks()
        except tk.TclError:
            pass  # Ignorar errores si la ventana ya no existe
    
    def safe_destroy(self):
        """Destruye la ventana de forma segura"""
        try:
            if self.winfo_exists():
                self.destroy()
        except tk.TclError:
            pass
class VentanaVisualizacion(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Pronóstico de Temperatura y Confianza")
        self.geometry("1000x600")
        self.minsize(800, 400)
        
        # Centrar la ventana
        self.center_window()
        
        # Crear frames principales
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pred_frame = ttk.Frame(self)
        self.pred_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configurar barra de herramientas
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Botones de control
        self.create_toolbar()
        
        # Configurar eventos de ventana
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables de estado
        self.current_view = "temperature"  # Valor por defecto
        
    def center_window(self):
        """Centra la ventana en la pantalla"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
    def create_toolbar(self):
        """Crea la barra de herramientas con controles"""
        # Botón para alternar vista
        self.toggle_btn = ttk.Button(
            self.toolbar_frame,
            text="Cambiar Vista",
            command=self.toggle_view
        )
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # Botón para exportar
        self.export_btn = ttk.Button(
            self.toolbar_frame,
            text="Exportar Gráfica",
            command=self.export_graph
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Botón para actualizar
        self.refresh_btn = ttk.Button(
            self.toolbar_frame,
            text="Actualizar",
            command=self.refresh_view
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
    def toggle_view(self):
        """Alterna entre diferentes vistas de visualización"""
        self.current_view = "confidence" if self.current_view == "temperature" else "temperature"
        self.refresh_view()
        
    def export_graph(self):
        """Exporta la gráfica actual como imagen"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                for widget in self.main_frame.winfo_children():
                    if isinstance(widget, FigureCanvasTkAgg):
                        widget.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                        messagebox.showinfo("Éxito", "Gráfica exportada exitosamente")
                        break
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar la gráfica: {str(e)}")
            
    def refresh_view(self):
        """Actualiza la vista actual"""
        # Esta función será llamada desde la clase principal
        pass
    def actualizar_graficas_iniciales(self, figuras):
        """Actualiza las gráficas iniciales"""
        try:
            # Limpiar frames
            for widget in self.main_frame.winfo_children():
                widget.destroy()
            for widget in self.pred_frame.winfo_children():
                widget.destroy()
                
            # Mostrar gráfica de series temporales
            if 'series_temporal' in figuras:
                canvas_series = FigureCanvasTkAgg(figuras['series_temporal'], self.main_frame)
                canvas_series.draw()
                canvas_series.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Agregar barra de navegación
                toolbar_series = NavigationToolbar2Tk(canvas_series, self.pred_frame)
                toolbar_series.update()
                
            # Mostrar gráfica de distribución
            if 'distribucion' in figuras:
                canvas_dist = FigureCanvasTkAgg(figuras['distribucion'], self.main_frame)
                canvas_dist.draw()
                canvas_dist.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Agregar barra de navegación
                toolbar_dist = NavigationToolbar2Tk(canvas_dist, self.pred_frame)
                toolbar_dist.update()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al actualizar las gráficas iniciales: {str(e)}")
    def actualizar_grafica(self, predicciones, visualizador):
        """Actualiza la gráfica con nuevas predicciones"""
        try:
            # Limpiar frames
            for widget in self.main_frame.winfo_children():
                widget.destroy()
            for widget in self.pred_frame.winfo_children():
                widget.destroy()
                
            # Crear nueva figura según la vista actual
            if self.current_view == "temperature":
                fig = visualizador.crear_grafica_temperatura(predicciones)
            else:
                fig = visualizador.crear_grafica_confianza(predicciones)
                
            # Mostrar gráfica principal
            canvas = FigureCanvasTkAgg(fig, self.main_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Agregar barra de navegación
            toolbar = NavigationToolbar2Tk(canvas, self.pred_frame)
            toolbar.update()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al actualizar la gráfica: {str(e)}")
            
    def on_closing(self):
        """Maneja el cierre de la ventana"""
        self.destroy()

class VentanaPronosticoDetallado(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Pronóstico Detallado por Períodos")
        self.geometry("1000x600")
        self.minsize(800, 400)
        
        # Centrar la ventana
        self.center_window()
        
        # Crear frames principales
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.controls_frame = ttk.Frame(self)
        self.controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Crear widgets de control
        self.create_controls()
        
        # Variables de estado
        self.current_period = "all"  # Valores: "all", "day", "night"
        self.show_confidence = True
        
        # Configurar eventos
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def center_window(self):
        """Centra la ventana en la pantalla"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
    def create_controls(self):
        """Crea los controles de la interfaz"""
        # Frame para filtros
        filter_frame = ttk.LabelFrame(self.controls_frame, text="Filtros")
        filter_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Combobox para período
        ttk.Label(filter_frame, text="Período:").pack(side=tk.LEFT, padx=5)
        self.period_var = tk.StringVar(value="all")
        period_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.period_var,
            values=["Todos", "Día", "Noche"],
            state="readonly",
            width=10
        )
        period_combo.pack(side=tk.LEFT, padx=5)
        period_combo.bind('<<ComboboxSelected>>', self.on_period_change)
        
        # Checkbox para mostrar confianza
        self.confidence_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            filter_frame,
            text="Mostrar Confianza",
            variable=self.confidence_var,
            command=self.on_confidence_toggle
        ).pack(side=tk.LEFT, padx=5)
        
        # Frame para botones
        button_frame = ttk.Frame(self.controls_frame)
        button_frame.pack(side=tk.RIGHT, padx=5)
        
        # Botón de exportar
        ttk.Button(
            button_frame,
            text="Exportar",
            command=self.export_forecast
        ).pack(side=tk.RIGHT, padx=5)
        
        # Botón de actualizar
        ttk.Button(
            button_frame,
            text="Actualizar",
            command=self.refresh_view
        ).pack(side=tk.RIGHT, padx=5)
        
    def on_period_change(self, event):
        """Maneja el cambio de período"""
        selection = self.period_var.get()
        if selection == "Todos":
            self.current_period = "all"
        elif selection == "Día":
            self.current_period = "day"
        else:
            self.current_period = "night"
        self.refresh_view()
        
    def on_confidence_toggle(self):
        """Maneja el cambio en la visualización de confianza"""
        self.show_confidence = self.confidence_var.get()
        self.refresh_view()
        
    def export_forecast(self):
        """Exporta el pronóstico detallado"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                for widget in self.main_frame.winfo_children():
                    if isinstance(widget, FigureCanvasTkAgg):
                        widget.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                        messagebox.showinfo("Éxito", "Pronóstico exportado exitosamente")
                        break
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar el pronóstico: {str(e)}")
            
    def refresh_view(self):
        """Actualiza la vista actual"""
        # Esta función será llamada desde la clase principal
        pass
        
    def actualizar_grafica(self, predicciones, visualizador):
        """Actualiza la gráfica con nuevas predicciones"""
        try:
            # Limpiar frames
            for widget in self.main_frame.winfo_children():
                widget.destroy()
                    
            # Crear nueva figura usando el visualizador
            fig = visualizador.crear_grafica_pronostico_detallado(predicciones)  # Removido el show_confidence
                    
            # Mostrar gráfica principal
            canvas = FigureCanvasTkAgg(fig, self.main_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            # Agregar barra de navegación
            toolbar = NavigationToolbar2Tk(canvas, self.pred_frame)
            toolbar.update()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al actualizar la gráfica: {str(e)}")
            
    def filtrar_predicciones(self, predicciones):
        """Filtra las predicciones según el período seleccionado"""
        if self.current_period == "all":
            return predicciones
            
        predicciones_filtradas = []
        for pred in predicciones:
            hora = int(pred['hora'].split(':')[0])
            if self.current_period == "day" and 6 <= hora <= 18:
                predicciones_filtradas.append(pred)
            elif self.current_period == "night" and (hora < 6 or hora > 18):
                predicciones_filtradas.append(pred)
                
        return predicciones_filtradas
        
    def on_closing(self):
        """Maneja el cierre de la ventana"""
        self.destroy()

class MicroClimaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predicción de Microclima de Facatativá")
        self.root.geometry("800x900")
        self.root.minsize(600, 800)
        
        # Centrar la ventana principal
        self.center_window()
        
        # Configurar grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Inicializar modelos y visualizadores
        self.predictor = PrediccionMicroclima()
        self.visualizador = VisualizacionMicroclima()
        
        # Inicializar variables de ventanas
        self.ventana_viz = None
        self.ventana_pronostico = None
        self.ventana_progreso = None
        # Definir pred_frame
        self.pred_frame = None  # O inicialízalo con el valor que necesites
        # Crear frame principal
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Variables de estado
        self.dataset = None
        self.ultimo_modelo = None
        self.entrenamiento_activo = False
        
        # Crear interfaz
        self.create_main_interface()
        
        # Configurar estilo
        self.configure_styles()
        
    def configure_styles(self):
        """Configura los estilos de la interfaz"""
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10))
        style.configure('Info.TLabel', font=('Arial', 9))
        
    def center_window(self):
        """Centra la ventana en la pantalla"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
    def create_main_interface(self):
        """Crea la interfaz principal"""
        main_panel = ttk.Frame(self.main_frame)
        main_panel.grid(row=0, column=0, sticky="nsew")
        main_panel.grid_columnconfigure(0, weight=1)
        
        # Crear secciones de la interfaz
        self.create_file_selection(main_panel)
        self.create_data_display(main_panel)
        self.create_training_area(main_panel)
        self.create_prediction_area(main_panel)
        self.create_visualization_area(main_panel)
    def create_file_selection(self, parent):
        """Crea la sección de selección de archivos"""
        file_frame = ttk.LabelFrame(parent, text="Selección de Datos", padding="5")
        file_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        file_frame.grid_columnconfigure(1, weight=1)
        
        # Botón de selección
        ttk.Button(file_frame, 
                  text="Seleccionar Dataset",
                  command=self.load_dataset, 
                  width=20).grid(row=0, column=0, padx=5, pady=5)
        
        # Etiqueta de archivo
        self.file_label = ttk.Label(file_frame, 
                                  text="Ningún archivo seleccionado", 
                                  wraplength=400, 
                                  style='Info.TLabel')
        self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
    def create_data_display(self, parent):
        """Crea la sección de visualización de datos"""
        data_frame = ttk.LabelFrame(parent, text="Información del Dataset", padding="5")
        data_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        data_frame.grid_columnconfigure(0, weight=1)
        
        # Área de texto con scroll
        self.data_text = tk.Text(data_frame, 
                                height=8, 
                                width=70, 
                                font=('Consolas', 9))
        scroll = ttk.Scrollbar(data_frame, 
                             orient="vertical", 
                             command=self.data_text.yview)
        self.data_text.configure(yscrollcommand=scroll.set)
        
        self.data_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        scroll.grid(row=0, column=1, sticky="ns")
        
    def create_training_area(self, parent):
        """Crea la sección de entrenamiento"""
        train_frame = ttk.LabelFrame(parent, text="Entrenamiento del Modelo", padding="5")
        train_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Frame para botones
        button_frame = ttk.Frame(train_frame)
        button_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Botones de entrenamiento
        ttk.Button(button_frame, 
                  text="Cargar Modelo",
                  command=self.cargar_modelo, 
                  width=20).grid(row=0, column=0, padx=5)
        
        ttk.Button(button_frame, 
                  text="Entrenar Modelo",
                  command=self.train_model, 
                  width=20).grid(row=0, column=1, padx=20)
        
        # Frame para parámetros
        param_frame = ttk.Frame(train_frame)
        param_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Configuración de épocas
        ttk.Label(param_frame, text="Épocas:").grid(row=0, column=0, padx=5)
        self.epochs_var = tk.StringVar(value="200")
        ttk.Entry(param_frame, 
                 textvariable=self.epochs_var, 
                 width=8).grid(row=0, column=1, padx=5)
        
        # Configuración de batch size
        ttk.Label(param_frame, text="Batch Size:").grid(row=0, column=2, padx=5)
        self.batch_size_var = tk.StringVar(value="64")
        ttk.Entry(param_frame, 
                 textvariable=self.batch_size_var, 
                 width=8).grid(row=0, column=3, padx=5)
        
        # Log de entrenamiento
        self.train_log = tk.Text(train_frame, 
                                height=5, 
                                width=70, 
                                font=('Consolas', 9))
        scroll = ttk.Scrollbar(train_frame, 
                             orient="vertical", 
                             command=self.train_log.yview)
        self.train_log.configure(yscrollcommand=scroll.set)
        
        self.train_log.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        scroll.grid(row=2, column=1, sticky="ns")
    def create_prediction_area(self, parent):
        """Crea la sección de predicciones"""
        pred_frame = ttk.LabelFrame(parent, text="Predicciones", padding="5")
        pred_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Panel superior para mostrar clima actual
        current_weather_frame = ttk.Frame(pred_frame)
        current_weather_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Temperatura actual
        self.current_temp_label = ttk.Label(current_weather_frame, 
                                          text="--°C", 
                                          font=('Arial', 24, 'bold'))
        self.current_temp_label.grid(row=0, column=0, padx=10)
        
        # Información actual
        current_info_frame = ttk.Frame(current_weather_frame)
        current_info_frame.grid(row=0, column=1, sticky="w")
        
        self.current_condition_label = ttk.Label(current_info_frame, 
                                               text="--", 
                                               font=('Arial', 12))
        self.current_condition_label.grid(row=0, column=0, sticky="w")
        
        self.current_confidence_label = ttk.Label(current_info_frame, 
                                                text="Confianza: --%",
                                                font=('Arial', 10))
        self.current_confidence_label.grid(row=1, column=0, sticky="w")
        
        # Controles de predicción
        control_frame = ttk.Frame(pred_frame)
        control_frame.grid(row=1, column=0, padx=5, pady=5)
        
        # Botón de predicción
        self.predict_button = ttk.Button(control_frame, 
                                       text="Generar Predicciones",
                                       command=self.generate_predictions, 
                                       width=20)
        self.predict_button.grid(row=0, column=0, padx=5)
        
        # Botón de exportar predicciones
        self.export_button = ttk.Button(control_frame,
                                      text="Exportar Predicciones",
                                      command=self.export_predictions,
                                      width=20)
        self.export_button.grid(row=0, column=1, padx=5)
        
        # Área de texto para predicciones
        self.pred_text = tk.Text(pred_frame, 
                                height=8, 
                                width=70, 
                                font=('Consolas', 9))
        scroll = ttk.Scrollbar(pred_frame, 
                             orient="vertical", 
                             command=self.pred_text.yview)
        self.pred_text.configure(yscrollcommand=scroll.set)
        
        self.pred_text.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        scroll.grid(row=2, column=1, sticky="ns")
    def mostrar_visualizaciones(self, tipo="temperatura"):
        """Muestra las visualizaciones según el tipo seleccionado"""
        try:
            if tipo == "detallado":
                # Primero destruir la ventana anterior si existe
                if hasattr(self, 'ventana_pronostico'):
                    try:
                        if isinstance(self.ventana_pronostico, tk.Toplevel) and self.ventana_pronostico.winfo_exists():
                            self.ventana_pronostico.destroy()
                    except:
                        pass
                        
                # Crear nueva ventana
                self.ventana_pronostico = VentanaPronosticoDetallado(self.root)
                
                # Actualizar datos si existen predicciones
                if hasattr(self, 'predictor') and self.predictor.model is not None:
                    try:
                        predicciones = self.predictor.predecir_proximo_periodo(self.dataset)
                        self.ventana_pronostico.actualizar_grafica(predicciones, self.predictor.visualizador)
                    except Exception as e:
                        self.show_error("Error", f"Error al actualizar predicciones: {str(e)}")
                
                # Traer ventana al frente
                self.ventana_pronostico.lift()
                
        except Exception as e:
            self.show_error("Error", f"Error al mostrar visualización: {str(e)}")
            print(f"Error detallado: {str(e)}")  # Para debugging
    def create_visualization_area(self, parent):
        """Crea la sección de visualizaciones"""
        viz_frame = ttk.LabelFrame(parent, text="Visualizaciones", padding="5")
        viz_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        button_frame = ttk.Frame(viz_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botón de visualización de temperatura
        ttk.Button(button_frame, 
                  text="Ver Temperatura",
                  command=lambda: self.mostrar_visualizaciones("temperatura"), 
                  width=20).pack(side=tk.LEFT, padx=5)
        
        # Botón de visualización detallada
        ttk.Button(button_frame, 
                  text="Ver Pronóstico Detallado",
                  command=lambda: self.mostrar_visualizaciones("detallado"), 
                  width=20).pack(side=tk.LEFT, padx=5)
    def cargar_modelo(self):
        """Carga un modelo previamente guardado"""
        try:
            self.ventana_progreso = VentanaProgreso(self.root, "Cargando Modelo")
            self.ventana_progreso.update_progress(0, "Iniciando carga del modelo...")
            
            def load_model():
                try:
                    self.predictor.cargar_modelo_guardado()
                    self.root.after(0, self.after_model_load)
                except Exception as e:
                    self.root.after(0, lambda: self.show_error("Error al cargar modelo", str(e)))
                finally:
                    if self.ventana_progreso:
                        self.ventana_progreso.safe_destroy()
            
            threading.Thread(target=load_model, daemon=True).start()
            
        except Exception as e:
            self.show_error("Error", f"Error al cargar el modelo: {str(e)}")
    def export_predictions(self):
        """Exporta las predicciones a un archivo CSV"""
        try:
            # Verificar si hay predicciones para exportar
            if not self.pred_text.get(1.0, tk.END).strip():
                messagebox.showwarning("Aviso", "No hay predicciones para exportar")
                return
            
            # Abrir diálogo para guardar archivo
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
                title="Exportar Predicciones"
            )
            
            if filename:
                # Mostrar ventana de progreso
                self.ventana_progreso = VentanaProgreso(self.root, "Exportando Predicciones")
                self.ventana_progreso.update_progress(0, "Preparando datos...")
                
                def export():
                    try:
                        # Obtener las últimas predicciones
                        predicciones = self.predictor.predecir_proximo_periodo(self.dataset)
                        
                        # Crear DataFrame con las predicciones
                        data = []
                        for pred in predicciones:
                            data.append({
                                'Fecha': pred['fecha'],
                                'Hora': pred['hora'],
                                'Categoría': pred['categoria'],
                                'Temperatura': f"{pred['temperatura']:.1f}°C",
                                'Confianza': f"{pred['confianza']*100:.1f}%",
                                'Descripción': pred['detalles']['descripcion'],
                                'Recomendaciones': ', '.join(pred['detalles']['recomendaciones'])
                            })
                        
                        # Crear y guardar DataFrame
                        df = pd.DataFrame(data)
                        df.to_csv(filename, index=False, encoding='utf-8')
                        
                        self.root.after(0, lambda: messagebox.showinfo(
                            "Éxito", 
                            "Predicciones exportadas exitosamente"
                        ))
                        
                    except Exception as e:
                        self.root.after(0, lambda: self.show_error(
                            "Error al exportar", 
                            str(e)
                        ))
                    finally:
                        if self.ventana_progreso:
                            self.ventana_progreso.safe_destroy()
                
                # Ejecutar exportación en un hilo separado
                threading.Thread(target=export, daemon=True).start()
                
        except Exception as e:
            self.show_error("Error", f"Error al exportar predicciones: {str(e)}")        


    def after_model_load(self):
        """Acciones después de cargar el modelo"""
        self.train_log.delete(1.0, tk.END)
        self.train_log.insert(tk.END, "Modelo cargado exitosamente\n")
        self.predict_button.config(state='normal')
        messagebox.showinfo("Éxito", "Modelo cargado exitosamente")
    def update_training_progress(self, epoch, total_epochs):
        """Actualiza el progreso del entrenamiento en la interfaz"""
        try:
            if self.ventana_progreso and self.ventana_progreso.winfo_exists():
                # Calcular el progreso actual
                progress = ((epoch + 1) / total_epochs) * 100
                
                # Actualizar la ventana de progreso
                self.ventana_progreso.update_progress(
                    progress,
                    f"Entrenando época {epoch + 1}/{total_epochs}"
                )
                
                # Actualizar el log de entrenamiento
                self.root.after(0, lambda: self.train_log.insert(tk.END, 
                    f"\nProgreso: {progress:.1f}% - Época {epoch + 1}/{total_epochs}"))
                self.root.after(0, lambda: self.train_log.see(tk.END))
                
                # Forzar actualización de la interfaz
                self.root.update_idletasks()
        except Exception as e:
            print(f"Error actualizando progreso: {str(e)}")
            if self.ventana_progreso:
                try:
                    self.ventana_progreso.safe_destroy()
                except:
                    pass
    def load_dataset(self):
        """Carga el dataset seleccionado"""
        try:
            filename = filedialog.askopenfilename(
                title="Seleccionar Dataset",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )
            if filename:
                self.file_label.config(text=os.path.basename(filename))
                self.ventana_progreso = VentanaProgreso(self.root, "Cargando Dataset")
                self.ventana_progreso.update_progress(0, "Iniciando carga...")
                
                def load_data():
                    try:
                        self.dataset = self.predictor.cargar_datos(filename)
                        self.root.after(0, self.after_dataset_load)
                    except Exception as e:
                        self.root.after(0, lambda: self.show_error("Error al cargar datos", str(e)))
                    finally:
                        if self.ventana_progreso:
                            self.ventana_progreso.safe_destroy()
                
                threading.Thread(target=load_data, daemon=True).start()
                
        except Exception as e:
            self.show_error("Error", f"Error al seleccionar archivo: {str(e)}")
    
    def after_dataset_load(self):
        """Acciones después de cargar el dataset"""
        self.show_dataset_info()
        self.enable_training_controls()
        self.predict_button.config(state='normal')
        
    def show_dataset_info(self):
        """Muestra información del dataset y visualizaciones"""
        try:
            # Limpiar y mostrar información básica
            self.data_text.delete(1.0, tk.END)
            info = f"Registros totales: {len(self.dataset)}\n"
            info += f"Rango de fechas: {str(self.dataset.index.min())} a {str(self.dataset.index.max())}\n\n"
            
            # Añadir resumen de variables
            info += "Resumen de variables:\n"
            for column in self.dataset.columns:
                info += f"\n{column}:\n"
                try:
                    info += f"  Min: {self.dataset[column].min()}\n"
                    info += f"  Max: {self.dataset[column].max()}\n"
                    info += f"  Media: {self.dataset[column].mean()}\n"
                    info += f"  Desv. Est.: {self.dataset[column].std()}\n"
                    info += f"  Valores nulos: {self.dataset[column].isnull().sum()}\n"
                except:
                    info += "  No se pueden calcular estadísticas para esta variable\n"
            
            # Añadir información de correlaciones
            info += "\nMatriz de correlación:\n"
            try:
                correlaciones = self.dataset.corr().round(2)
                info += str(correlaciones)
            except:
                info += "No se pueden calcular las correlaciones\n"
            
            info += "\nEstadísticas descriptivas completas:\n"
            try:
                desc = self.dataset.describe().round(2)
                info += str(desc)
            except:
                info += "No se pueden calcular las estadísticas descriptivas completas\n"
            
            self.data_text.insert(tk.END, info)

            # Crear visualización de series temporales
            try:
                # Inicializar predictor si no existe
                if not hasattr(self, 'predictor'):
                    self.predictor = PrediccionMicroclima()
                    
                # Crear visualizador y asignarlo al predictor
                self.predictor.visualizador = VisualizacionMicroclima()
                
                # Asegurarse de que la ventana_viz sea None si no existe
                if hasattr(self, 'ventana_viz'):
                    try:
                        if not self.ventana_viz.winfo_exists():
                            self.ventana_viz = None
                    except:
                        self.ventana_viz = None
                
                # Crear nueva ventana si no existe
                if not hasattr(self, 'ventana_viz') or self.ventana_viz is None:
                    self.ventana_viz = VentanaVisualizacion(self.root)
                
                # Crear diferentes visualizaciones usando el visualizador del predictor
                fig_series = self.predictor.visualizador.plot_series_temporal(self.dataset)
                fig_dist = self.predictor.visualizador.plot_distribucion_condiciones(self.dataset)
                
                # Crear directorio temporal si no existe
                if not os.path.exists('temp_figures'):
                    os.makedirs('temp_figures')
                
                # Guardar figuras temporalmente
                try:
                    fig_series.savefig('temp_figures/series_temporal.png')
                    fig_dist.savefig('temp_figures/distribucion.png')
                except Exception as e:
                    print(f"Error al guardar figuras: {e}")
                
                # Actualizar visualizaciones en la ventana
                if hasattr(self.ventana_viz, 'actualizar_graficas_iniciales'):
                    self.ventana_viz.actualizar_graficas_iniciales({
                        'series_temporal': fig_series,
                        'distribucion': fig_dist
                    })
                
                # Traer ventana al frente
                self.ventana_viz.lift()
                
                # Habilitar controles relevantes
                self.enable_training_controls()
                
                # Limpiar memoria de las figuras
                plt.close(fig_series)
                plt.close(fig_dist)
                
            except Exception as e:
                self.show_error("Error en visualización", 
                            f"Error al mostrar visualizaciones: {str(e)}")
                print(f"Error detallado en visualización: {str(e)}")
                
        except Exception as e:
            self.show_error("Error", 
                        f"Error al mostrar información del dataset: {str(e)}")
            print(f"Error detallado: {str(e)}")
            
        finally:
            # Limpiar archivos temporales antiguos
            try:
                if os.path.exists('temp_figures'):
                    for file in os.listdir('temp_figures'):
                        if file.endswith('.png'):
                            file_path = os.path.join('temp_figures', file)
                            if os.path.getctime(file_path) < time.time() - 3600:  # Más de 1 hora
                                os.remove(file_path)
            except Exception as e:
                print(f"Error al limpiar archivos temporales: {e}")
            
            # Asegurar que el cursor vuelva al inicio del texto
            self.data_text.see("1.0")
            
            # Actualizar la interfaz
            self.root.update_idletasks()
            
            # Limpiar memoria
            gc.collect()
    def train_model(self):
        """Inicia el entrenamiento del modelo"""
        if not hasattr(self, 'dataset'):
            self.show_error("Error", "Primero debe cargar un dataset")
            return
                
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            
            self.train_log.delete(1.0, tk.END)
            self.train_log.insert(tk.END, "Iniciando entrenamiento...\n")
            
            self.ventana_progreso = VentanaProgreso(self.root, "Entrenando Modelo")
            self.entrenamiento_activo = True

            def handle_error(error_msg):
                """Maneja los errores de entrenamiento"""
                def show_error():
                    self.show_error("Error en entrenamiento", error_msg)
                    if self.ventana_progreso:
                        self.ventana_progreso.safe_destroy()
                self.root.after(0, show_error)

            def handle_success(history):
                """Maneja el éxito del entrenamiento"""
                def update_ui():
                    self.update_training_results(history)
                    if self.ventana_progreso:
                        self.ventana_progreso.safe_destroy()
                    
                    # Mostrar mensaje de éxito
                    messagebox.showinfo("Éxito", "Entrenamiento completado exitosamente")
                    
                    # Habilitar botones de predicción
                    self.predict_button.config(state='normal')
                    
                self.root.after(0, update_ui)

            def train():
                try:
                    print("Iniciando hilo de entrenamiento...")
                    print("Preparando datos para entrenamiento...")
                    print(f"Iniciando entrenamiento con {epochs} épocas y batch size de {batch_size}")
                    
                    history = self.predictor.entrenar_modelo(
                        df=self.dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        callback=self.update_training_progress
                    )
                    
                    handle_success(history)
                    
                except Exception as e:
                    handle_error(str(e))
                finally:
                    self.entrenamiento_activo = False
                    print("Entrenamiento finalizado")
            
            # Iniciar entrenamiento en un hilo separado
            training_thread = threading.Thread(target=train, daemon=True)
            training_thread.start()
            
        except ValueError:
            self.show_error("Error", "Los valores de épocas y batch size deben ser números enteros")
        except Exception as e:
            self.show_error("Error", f"Error al iniciar entrenamiento: {str(e)}")
            if self.ventana_progreso:
                self.ventana_progreso.safe_destroy()
    
    def update_training_results(self, history):
        """Actualiza los resultados del entrenamiento en la interfaz"""
        try:
            # Limpiar log anterior
            self.train_log.delete(1.0, tk.END)
            self.train_log.insert(tk.END, "Entrenamiento completado\n\n")
            
            # Verificar si history es None
            if history is None:
                self.train_log.insert(tk.END, "No hay métricas disponibles del entrenamiento\n")
                self.predict_button.config(state='disabled')
                messagebox.showwarning("Advertencia", "El entrenamiento no generó métricas")
                return
            
            # Verificar si history tiene el atributo history
            if not hasattr(history, 'history'):
                self.train_log.insert(tk.END, "No se obtuvieron métricas del entrenamiento\n")
                self.predict_button.config(state='disabled')
                messagebox.showwarning("Advertencia", "No se obtuvieron métricas del entrenamiento")
                return
            
            # Mostrar métricas finales
            self.train_log.insert(tk.END, "Métricas finales:\n")
            metrics = history.history
            
            # Actualizar métricas en la interfaz
            try:
                if 'accuracy' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Precisión de entrenamiento: {metrics['accuracy'][-1]:.4f}\n")
                if 'loss' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Pérdida de entrenamiento: {metrics['loss'][-1]:.4f}\n")
                if 'val_accuracy' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Precisión de validación: {metrics['val_accuracy'][-1]:.4f}\n")
                if 'val_loss' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Pérdida de validación: {metrics['val_loss'][-1]:.4f}\n")
                
                # Métricas adicionales si están disponibles
                if 'precision' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Precisión: {metrics['precision'][-1]:.4f}\n")
                if 'recall' in metrics:
                    self.train_log.insert(tk.END, 
                        f"Recall: {metrics['recall'][-1]:.4f}\n")
                if 'auc' in metrics:
                    self.train_log.insert(tk.END, 
                        f"AUC: {metrics['auc'][-1]:.4f}\n")
            except Exception as e:
                self.train_log.insert(tk.END, f"\nError al mostrar algunas métricas: {str(e)}\n")
            
            # Habilitar botones de predicción
            self.predict_button.config(state='normal')
            
            # Mostrar mensaje de éxito
            messagebox.showinfo("Éxito", "Entrenamiento completado exitosamente")
            
            # Actualizar interfaz
            self.root.update_idletasks()
            
        except Exception as e:
            self.show_error("Error", f"Error al actualizar resultados: {str(e)}")
        finally:
            # Asegurar que la ventana de progreso se cierre
            if self.ventana_progreso:
                self.ventana_progreso.safe_destroy()
            
            # Limpiar memoria
            gc.collect()
            
    def generate_predictions(self):
        """Genera predicciones con el modelo actual"""
        if not hasattr(self, 'dataset'):
            self.show_error("Error", "Primero debe cargar un dataset")
            return
            
        try:
            self.ventana_progreso = VentanaProgreso(self.root, "Generando Predicciones")
            
            def predict():
                try:
                    predicciones = self.predictor.predecir_proximo_periodo(self.dataset)
                    self.root.after(0, lambda: self.update_predictions(predicciones))
                except Exception as e:
                    self.root.after(0, lambda: self.show_error("Error en predicción", str(e)))
                finally:
                    if self.ventana_progreso:
                        self.ventana_progreso.safe_destroy()
            
            threading.Thread(target=predict, daemon=True).start()
            
        except Exception as e:
            self.show_error("Error", f"Error al generar predicciones: {str(e)}")
            
    def update_predictions(self, predicciones):
        """Actualiza la interfaz con las nuevas predicciones"""
        # Actualizar información actual
        primera_pred = predicciones[0]
        self.current_temp_label.config(text=f"{primera_pred['temperatura']:.1f}°C")
        self.current_condition_label.config(text=primera_pred['categoria'])
        self.current_confidence_label.config(text=f"Confianza: {primera_pred['confianza']*100:.1f}%")
        
        # Actualizar texto de predicciones
        self.pred_text.delete(1.0, tk.END)
        for pred in predicciones:
            self.pred_text.insert(tk.END, 
                f"{pred['fecha']}: {pred['categoria']} "
                f"(Confianza: {pred['confianza']*100:.1f}%)\n"
                f"Temperatura: {pred['temperatura']:.1f}°C\n"
                f"{pred['detalles']['descripcion']}\n"
                f"Recomendaciones: {', '.join(pred['detalles']['recomendaciones'])}\n\n")
        
        # Actualizar visualizaciones si están abiertas
        if self.ventana_viz and self.ventana_viz.winfo_exists():
            self.ventana_viz.actualizar_grafica(predicciones, self.visualizador)
        if self.ventana_pronostico and self.ventana_pronostico.winfo_exists():
            self.ventana_pronostico.actualizar_grafica(predicciones, self.visualizador)
            
    def show_error(self, title, message):
        """Muestra un mensaje de error"""
        messagebox.showerror(title, message)
        
    def enable_training_controls(self):
        """Habilita los controles de entrenamiento"""
        for child in self.main_frame.winfo_children():
            if isinstance(child, ttk.Button):
                child.config(state='normal')

if __name__ == "__main__":
    try:
        print("Iniciando aplicación...")
        root = tk.Tk()
        app = MicroClimaGUI(root)
        print("GUI inicializada")
        root.mainloop()
    except Exception as e:
        print(f"Error al iniciar la aplicación: {str(e)}")