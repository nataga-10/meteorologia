import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os


# Configuración global de matplotlib
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.autolayout'] = False

class VisualizacionMicroclima:
    def __init__(self):
        # Configurar estilo general
        sns.set_theme(style="whitegrid")
        plt.style.use('default')
        
        # Paleta de colores actualizada para las variables
        self.colores = {
            'temperatura_C': '#FF9671',
            'humedad_relativa': '#00D4FF',
            'precipitacion_mm': '#005EFF',
            'cobertura_nubes_octas': '#A5A5A5',
            'velocidad_viento_kmh': '#69B34C',
            'radiacion_solar_J_m2': '#FFD700'
        }
        
        # Paleta de colores para categorías
        self.paleta_categorias = {
            'Frío': '#00B4D8',
            'Templado': '#90BE6D',
            'Cálido': '#F94144',
            'Muy Húmedo': '#277DA1',
            'Húmedo': '#4D908E',
            'Lluvia Fuerte': '#577590',
            'Llovizna': '#43AA8B',
            'Muy Nublado': '#758E4F',
            'Parcialmente Nublado': '#F9C74F',
            'Normal': '#F8961E'
        }
        
        # Rutas de las imágenes para condiciones climáticas
        self.weather_icons = {
            'Frío': 'Imagenes-Clima/Frio.png',
            'Lluvia Fuerte': 'Imagenes-Clima/Fuerte_Lluvia.png',
            'Llovizna': 'Imagenes-Clima/Lluvia.png',
            'Muy Nublado': 'Imagenes-Clima/Nublado.png',
            'Parcialmente Nublado': 'Imagenes-Clima/Parcialmente_Soleado.png',
            'Normal': 'Imagenes-Clima/Soleado.png'
        }
    def configurar_estilo_grafica(self, ax, titulo, xlabel, ylabel):
        """Configura el estilo común para todas las gráficas"""
        ax.set_title(titulo, fontsize=12, pad=20)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    def get_weather_icon(self, categoria):
        """Carga y devuelve la imagen correspondiente a la categoría"""
        for key, icon_path in self.weather_icons.items():
            if key in categoria:
                try:
                    if os.path.exists(icon_path):
                        img = plt.imread(icon_path)
                        # Asegurarnos de que la imagen tenga el formato correcto
                        if len(img.shape) == 3 and img.shape[2] == 4:  # Si tiene canal alpha
                            # Normalizar valores de la imagen si es necesario
                            if img.max() > 1.0:
                                img = img / 255.0
                            return img
                        elif len(img.shape) == 3 and img.shape[2] == 3:  # Si es RGB
                            # Normalizar valores de la imagen si es necesario
                            if img.max() > 1.0:
                                img = img / 255.0
                            return img
                        else:
                            print(f"Formato de imagen no soportado para {key}: {icon_path}")
                            print(f"Forma de la imagen: {img.shape}")
                            return None
                    else:
                        print(f"No se encontró la imagen: {icon_path}")
                except Exception as e:
                    print(f"Error cargando imagen {icon_path} para {key}: {str(e)}")
                    return None
        return None

    def crear_grafica_temperatura(self, predicciones):
        """Crea una visualización del pronóstico de temperatura y confianza"""
        fig = Figure(figsize=(12, 6))
        fig.subplots_adjust(top=0.95, bottom=0.15)
        
        ax1 = fig.add_subplot(111)
        
        # Convertir predicciones a DataFrame
        df_pred = pd.DataFrame(predicciones)
        df_pred['fecha'] = pd.to_datetime(df_pred['fecha'])
        
        # Plotear temperatura
        ax1.plot(df_pred['fecha'], df_pred['temperatura'], 
                color=self.colores['temperatura_C'], 
                marker='o', 
                linewidth=2,
                label='Temperatura')
        
        # Agregar etiquetas de hora
        for idx, fecha in enumerate(df_pred['fecha']):
            if idx % 6 == 0:  # Mostrar cada 6 horas para no saturar
                ax1.annotate(fecha.strftime('%H:%M'),
                        (fecha, df_pred['temperatura'].iloc[idx]),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8)
        
        # Agregar área de confianza
        ax1_twin = ax1.twinx()
        confianza = df_pred['confianza'].values
        ax1_twin.fill_between(df_pred['fecha'], 0, confianza, 
                            alpha=0.2, 
                            color='green',
                            label='Nivel de Confianza')
        
        self.configurar_estilo_grafica(ax1, 'Pronóstico de Temperatura y Confianza',
                                    '', 'Temperatura (°C)')
        ax1_twin.set_ylabel('Nivel de Confianza')
        
        # Formatear eje x para mostrar fecha y hora
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Combinar leyendas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        fig.align_labels()
        return fig

    def crear_grafica_pronostico_detallado(self, predicciones):
        """Crea una visualización del pronóstico detallado por períodos"""
        # Configuración inicial de la figura
        fig = Figure(figsize=(12, 6))
        fig.subplots_adjust(top=0.92, bottom=0.15, left=0.12, right=0.88)
        
        ax = fig.add_subplot(111)
        
        # Convertir predicciones a DataFrame
        df_pred = pd.DataFrame(predicciones)
        df_pred['fecha'] = pd.to_datetime(df_pred['fecha'])
        
        # Agrupar predicciones por día y período
        df_pred['periodo'] = pd.cut(df_pred['fecha'].dt.hour,
                                bins=[0, 6, 12, 18, 24],
                                labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'])
        
        periodos = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
        y_pos = range(len(periodos))
        
        fechas_unicas = df_pred['fecha'].dt.date.unique()
        
        # Reducir el tamaño de los rectángulos para más espacio
        rect_width = 0.8
        rect_height = 0.6
        
        # Matriz de visualización
        for idx, fecha in enumerate(fechas_unicas):
            day_data = df_pred[df_pred['fecha'].dt.date == fecha]
            
            for i, periodo in enumerate(periodos):
                periodo_data = day_data[day_data['periodo'] == periodo]
                if not periodo_data.empty:
                    # Color según confianza
                    confianza = periodo_data['confianza'].mean()
                    color = plt.cm.RdYlGn(confianza)
                    
                    # Crear rectángulo centrado
                    rect = patches.Rectangle(
                        (idx - rect_width/2, i - rect_height/2),
                        rect_width, rect_height,
                        facecolor=color,
                        alpha=0.7,
                        edgecolor='gray',
                        linewidth=0.5
                    )
                    ax.add_patch(rect)
                    
                    # Agregar imagen del clima dentro del rectángulo
                    categoria = periodo_data['categoria'].iloc[0]
                    img = self.get_weather_icon(categoria)
                    if img is not None:
                        # Tamaño del ícono relativo al rectángulo
                        icon_size = min(rect_width, rect_height) * 0.8
                        
                        # Convertir coordenadas de datos a coordenadas de pantalla
                        x_center = idx
                        y_center = i
                        
                        # Agregar el ícono como una imagen en el eje principal
                        ax.imshow(img, 
                                extent=[x_center - icon_size/2, 
                                    x_center + icon_size/2,
                                    y_center - icon_size/2, 
                                    y_center + icon_size/2],
                                zorder=2)  # Asegurar que esté sobre el rectángulo
                    
                    # Posicionar textos
                    hora_texto = periodo_data['fecha'].iloc[0].strftime('%H:%M')
                    
                    # Porcentaje arriba del rectángulo
                    ax.text(idx, i + rect_height/2 + 0.1,
                        f"{confianza*100:.0f}%",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        zorder=3)  # Asegurar que esté sobre todo
                    
                    # Hora debajo del rectángulo
                    ax.text(idx, i - rect_height/2 - 0.1,
                        hora_texto,
                        ha='center',
                        va='top',
                        fontsize=8,
                        zorder=3)
        
        # Configurar ejes
        ax.set_yticks(y_pos)
        ax.set_yticklabels(periodos)
        ax.set_xticks(range(len(fechas_unicas)))
        ax.set_xticklabels([f.strftime('%d/%m') for f in fechas_unicas],
                        rotation=45,
                        ha='right')
        
        # Ajustar límites de los ejes
        ax.set_ylim(-0.5, len(periodos) - 0.5)
        ax.set_xlim(-0.5, len(fechas_unicas) - 0.5)
        
        self.configurar_estilo_grafica(ax, 'Pronóstico Detallado por Períodos',
                                    'Fecha', '')
        
        # Agregar barra de color para nivel de confianza
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Nivel de Confianza')
        
        fig.align_labels()
        return fig

    def plot_series_temporal(self, df):
        """Genera gráfico de series temporales para todas las variables"""
        fig = Figure(figsize=(12, 15))
        
        # Agregar más espacio para los subplots
        fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.1, left=0.1, right=0.9)
        
        # Crear subfiguras con espacio entre ellas
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 0.8, 1, 1], figure=fig)
        
        # 1. Gráfico de Temperatura y Humedad
        ax1 = fig.add_subplot(gs[0])
        
        # Plotear temperatura
        line_temp = ax1.plot(df.index, df['temperatura_C'], 
                            label='Temperatura', 
                            color=self.colores['temperatura_C'],
                            linewidth=1.5)
        ax1.set_ylabel('Temperatura (°C)', color=self.colores['temperatura_C'])
        ax1.tick_params(axis='y', labelcolor=self.colores['temperatura_C'])
        
        # Crear eje gemelo para humedad
        ax1_twin = ax1.twinx()
        line_hum = ax1_twin.plot(df.index, df['humedad_relativa'],
                                label='Humedad',
                                color=self.colores['humedad_relativa'],
                                linewidth=1.5)
        ax1_twin.set_ylabel('Humedad (%)', color=self.colores['humedad_relativa'])
        ax1_twin.tick_params(axis='y', labelcolor=self.colores['humedad_relativa'])
        
        # Combinar leyendas
        lines = line_temp + line_hum
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        ax1.set_title('Temperatura y Humedad', pad=20)
        ax1.grid(True, alpha=0.3)
        # 2. Gráfico de Precipitación
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df.index, df['precipitacion_mm'],
                color=self.colores['precipitacion_mm'],
                linewidth=1.5,
                label='Precipitación')
        ax2.set_ylabel('Precipitación (mm)')
        ax2.fill_between(df.index, df['precipitacion_mm'], 
                        color=self.colores['precipitacion_mm'], 
                        alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_title('Precipitación', pad=20)
        ax2.grid(True, alpha=0.3)
        
        # 3. Gráfico de Cobertura de Nubes y Viento
        ax3 = fig.add_subplot(gs[2])
        
        # Plotear cobertura de nubes
        line_nubes = ax3.plot(df.index, df['cobertura_nubes_octas'],
                            label='Cobertura Nubes',
                            color=self.colores['cobertura_nubes_octas'],
                            linewidth=1.5)
        ax3.set_ylabel('Cobertura de Nubes (octas)', 
                    color=self.colores['cobertura_nubes_octas'])
        ax3.tick_params(axis='y', labelcolor=self.colores['cobertura_nubes_octas'])
        
        # Crear eje gemelo para viento
        ax3_twin = ax3.twinx()
        line_viento = ax3_twin.plot(df.index, df['velocidad_viento_kmh'],
                                label='Velocidad Viento',
                                color=self.colores['velocidad_viento_kmh'],
                                linewidth=1.5)
        ax3_twin.set_ylabel('Velocidad del Viento (km/h)', 
                        color=self.colores['velocidad_viento_kmh'])
        ax3_twin.tick_params(axis='y', labelcolor=self.colores['velocidad_viento_kmh'])
        
        # Combinar leyendas para gráfico 3
        lines3 = line_nubes + line_viento
        labels3 = [l.get_label() for l in lines3]
        ax3.legend(lines3, labels3, loc='upper right')
        
        # 4. Gráfico de Radiación Solar
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(df.index, df['radiacion_solar_J_m2'],
                color=self.colores['radiacion_solar_J_m2'],
                linewidth=1.5,
                label='Radiación Solar')
        ax4.set_ylabel('Radiación Solar (J/m²)')
        ax4.fill_between(df.index, df['radiacion_solar_J_m2'], 
                        color=self.colores['radiacion_solar_J_m2'], 
                        alpha=0.3)
        ax4.legend(loc='upper right')
        ax4.set_title('Radiación Solar', pad=20)
        ax4.grid(True, alpha=0.3)
        
        # Configurar formato de fechas para todos los ejes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=45)
        
        fig.align_labels()  # Alinear las etiquetas
        return fig
    def plot_distribucion_condiciones(self, df):
        """Genera gráfico de distribución de condiciones climáticas"""
        fig = Figure(figsize=(12, 8))
        
        # Agregar más espacio para los subplots
        fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.15, left=0.1, right=0.9)
        
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], figure=fig)
        
        # Histograma de categorías
        ax1 = fig.add_subplot(gs[0])
        if 'categoria_clima' in df.columns:
            categoria_counts = df['categoria_clima'].value_counts()
            
            bars = ax1.bar(range(len(categoria_counts)), 
                          categoria_counts.values,
                          color=[self.paleta_categorias.get(cat, '#888888') 
                                for cat in categoria_counts.index],
                          alpha=0.7)
            
            ax1.set_xticks(range(len(categoria_counts)))
            ax1.set_xticklabels(categoria_counts.index, rotation=45, ha='right')
            
            # Agregar valores sobre las barras
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
            
            self.configurar_estilo_grafica(ax1, 'Distribución de Condiciones Climáticas',
                                         '', 'Frecuencia')
        
        # Boxplots de variables numéricas
        ax2 = fig.add_subplot(gs[1])
        variables = ['temperatura_C', 'humedad_relativa', 'precipitacion_mm', 
                    'radiacion_solar_J_m2']
        data_to_plot = [df[var].dropna() for var in variables]
        
        box = ax2.boxplot(data_to_plot,
                         patch_artist=True,
                         medianprops=dict(color="black"),
                         flierprops=dict(marker='o', markerfacecolor='gray'))
        
        # Colorear boxplots
        colors = [self.colores[var] for var in variables]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xticklabels(['Temperatura', 'Humedad', 'Precipitación', 'Radiación'],
                           rotation=45)
        self.configurar_estilo_grafica(ax2, 'Distribución de Variables',
                                     '', 'Valor')
        
        fig.align_labels()  # Alinear las etiquetas
        return fig
    def plot_metricas_entrenamiento(self, history):
        """Visualiza las métricas del entrenamiento del modelo"""
        fig = Figure(figsize=(12, 5))
        fig.subplots_adjust(wspace=0.3, top=0.9, bottom=0.15, left=0.1, right=0.9)
        
        gs = gridspec.GridSpec(1, 2, figure=fig)
        
        # Gráfico de pérdida
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(history.history['loss'], 
                label='Entrenamiento',
                color='#FF9671', 
                linewidth=2)
        ax1.plot(history.history['val_loss'], 
                label='Validación',
                color='#005EFF', 
                linewidth=2)
        
        self.configurar_estilo_grafica(ax1, 'Pérdida del Modelo', 
                                     'Época', 'Pérdida')
        ax1.legend()
        
        # Gráfico de precisión
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(history.history['accuracy'], 
                label='Entrenamiento',
                color='#FF9671', 
                linewidth=2)
        ax2.plot(history.history['val_accuracy'], 
                label='Validación',
                color='#005EFF', 
                linewidth=2)
        
        self.configurar_estilo_grafica(ax2, 'Precisión del Modelo', 
                                     'Época', 'Precisión')
        ax2.legend()
        
        fig.align_labels()  # Alinear las etiquetas
        return fig
    def crear_grafica_resumen_predicciones(self, predicciones):
        """Crea una visualización detallada de las predicciones"""
        fig = Figure(figsize=(14, 10))
        fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1)
        
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.5], figure=fig)
        
        # Convertir predicciones a DataFrame
        df_pred = pd.DataFrame(predicciones)
        df_pred['fecha'] = pd.to_datetime(df_pred['fecha'])
        
        # 1. Gráfico superior: Temperatura y confianza
        ax1 = fig.add_subplot(gs[0])
        
        # Plotear temperatura
        ax1.plot(df_pred['fecha'], df_pred['temperatura'], 
                color=self.colores['temperatura_C'], 
                marker='o', 
                linewidth=2,
                label='Temperatura')
        
        # Agregar etiquetas de hora
        for idx, fecha in enumerate(df_pred['fecha']):
            if idx % 6 == 0:  # Mostrar cada 6 horas para no saturar
                ax1.annotate(fecha.strftime('%H:%M'),
                        (fecha, df_pred['temperatura'].iloc[idx]),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8)
        
        # Agregar área de confianza
        ax1_twin = ax1.twinx()
        confianza = df_pred['confianza'].values
        ax1_twin.fill_between(df_pred['fecha'], 0, confianza, 
                            alpha=0.2, 
                            color='green',
                            label='Nivel de Confianza')
        
        self.configurar_estilo_grafica(ax1, 'Pronóstico de Temperatura y Confianza',
                                    '', 'Temperatura (°C)')
        ax1_twin.set_ylabel('Nivel de Confianza')
        
        # Formatear eje x para mostrar fecha y hora
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Combinar leyendas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 2. Gráfico medio: Categorías predominantes
        ax2 = fig.add_subplot(gs[1])
        categorias = df_pred['categoria'].values
        ax2.plot(df_pred['fecha'], categorias, 
                marker='o',
                linestyle='',
                color='blue')
        
        # Formatear eje x para mostrar fecha y hora
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylabel('Categorías')
        ax2.tick_params(axis='y', rotation=45)
        self.configurar_estilo_grafica(ax2, 'Categorías Predominantes', '', '')
        
        # 3. Gráfico inferior: Condiciones por período
        ax3 = fig.add_subplot(gs[2])
        
        # Agrupar predicciones por día y período
        df_pred['periodo'] = pd.cut(df_pred['fecha'].dt.hour,
                                bins=[0, 6, 12, 18, 24],
                                labels=['Madrugada', 'Mañana', 'Tarde', 'Noche'])
        
        periodos = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
        y_pos = range(len(periodos))
        
        fechas_unicas = df_pred['fecha'].dt.date.unique()
        
        # Matriz de visualización
        for idx, fecha in enumerate(fechas_unicas):
            day_data = df_pred[df_pred['fecha'].dt.date == fecha]
            
            for i, periodo in enumerate(periodos):
                periodo_data = day_data[day_data['periodo'] == periodo]
                if not periodo_data.empty:
                    # Color según confianza
                    confianza = periodo_data['confianza'].mean()
                    color = plt.cm.RdYlGn(confianza)
                    
                    # Crear rectángulo para el período
                    rect = patches.Rectangle(
                        (idx, i-0.4),
                        0.8, 0.8,
                        facecolor=color,
                        alpha=0.7
                    )
                    ax3.add_patch(rect)
                    
                    # Agregar imagen del clima
                    categoria = periodo_data['categoria'].iloc[0]
                    img = self.get_weather_icon(categoria)
                    if img is not None:
                        bbox = rect.get_bbox()
                        
                        # Tamaño fijo más pequeño
                        icon_size = 0.15
                        
                        # Ajustar posición para evitar superposición
                        icon_x = bbox.x0 + (bbox.width - icon_size) / 2
                        icon_y = bbox.y0 + (bbox.height - icon_size) / 2
                        
                        # Crear un identificador único para cada axes
                        icon_ax_name = f'icon_ax_{idx}_{i}'
                        
                        # Eliminar axes anterior si existe
                        for ax in fig.axes:
                            if ax.get_label() == icon_ax_name:
                                fig.delaxes(ax)
                        
                        # Crear nuevo axes con identificador único
                        icon_ax = fig.add_axes([
                            icon_x,
                            icon_y,
                            icon_size,
                            icon_size
                        ], label=icon_ax_name)
                        
                        # Configurar axes
                        icon_ax.set_frame_on(False)
                        icon_ax.imshow(img, aspect='equal')
                        icon_ax.axis('off')
                    
                    # Agregar texto de confianza y hora
                    hora_texto = periodo_data['fecha'].iloc[0].strftime('%H:%M')
                    ax3.text(idx + 0.4, i - 0.6,
                            f"{confianza*100:.0f}%\n{hora_texto}",
                            ha='center', 
                            va='top',
                            fontsize=8)
        
        # Configurar ejes
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(periodos)
        ax3.set_xticks(range(len(fechas_unicas)))
        ax3.set_xticklabels([f.strftime('%d/%m') for f in fechas_unicas],
                        rotation=45)
        
        self.configurar_estilo_grafica(ax3, 'Pronóstico Detallado por Períodos',
                                    'Fecha', '')
        
        # Agregar barra de color para nivel de confianza
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn)
        cbar = fig.colorbar(sm, ax=ax3)
        cbar.set_label('Nivel de Confianza')
        
        fig.align_labels()
        return fig

    def generar_reporte_visual(self, df, predicciones, history=None):
        """Genera un reporte visual completo"""
        fig = Figure(figsize=(15, 25))
        fig.subplots_adjust(hspace=0.4, top=0.95)
        
        gs = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1, 1, 1], figure=fig)
        
        # Series temporales
        ax1 = fig.add_subplot(gs[0])
        self.plot_series_temporal(df)
        
        # Distribución de condiciones
        ax2 = fig.add_subplot(gs[1])
        self.plot_distribucion_condiciones(df)
        
        # Predicciones
        ax3 = fig.add_subplot(gs[2])
        self.crear_grafica_resumen_predicciones(predicciones)
        
        # Métricas de entrenamiento (si están disponibles)
        if history is not None:
            ax4 = fig.add_subplot(gs[3])
            self.plot_metricas_entrenamiento(history)
        
        fig.suptitle('Reporte Completo de Análisis Climático',
                    fontsize=16, y=0.95)
        
        fig.align_labels()
        return fig

    def guardar_reporte(self, fig, nombre_archivo='reporte_climatico.png'):
        """Guarda el reporte visual en un archivo"""
        try:
            fig.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"Error al guardar el reporte: {str(e)}")
            return False