# **G-CLUS**
**Algoritmo de Clustering con Restricciones de Tamaño Basado en Grafos**

## _**Descripción**_
Este proyecto forma parte de mi tesis para la carrera de Ingeniería en Computación y tiene como objetivo desarrollar un algoritmo de clustering basado en grafos que aplique restricciones de tamaño específicas a los clusters formados. Este enfoque es crucial para aplicaciones prácticas donde la equidad y la gestión efectiva de recursos son fundamentales, como en la asignación de recursos, formación de equipos, organización de eventos y marketing social.

## _**Problema de Estudio**_
El estudio de la formación de grupos o clusters de instancias es sustancial para comprender la interacción y difusión de la información (Easley & Kleinberg, 2010). No obstante, los enfoques de clustering convencionales suelen generar grupos de tamaños muy variados, lo que puede restringir su utilidad en contextos prácticos, como la asignación equitativa de recursos, la formación de equipos, organización de eventos y marketing social (Zhang et al., 2024).

Por lo tanto, es necesario desarrollar algoritmos de clustering que no solo identifiquen segmentos o grupos en un determinado conjunto de datos, sino que también apliquen restricciones de tamaño a los clusters. Los grafos pueden ser empleados para representar relaciones complejas y restricciones en la agrupación de datos (Diestel, 2005). Este problema es relevante porque permite una distribución de grupos más equitativa y manejable, facilitando análisis y acciones posteriores más efectivas.

## _**Objetivos**_
### _**Objetivo General:**_
- Desarrollar un algoritmo de clustering con restricciones de tamaño basado en grafos.

### _**Objetivos Específicos:**_

- Analizar los algoritmos de clustering existentes, incluyendo la aplicación de la teoría de grafos y las restricciones de tamaño en su implementación.
- Definir un método de clustering para formar grupos de un tamaño fijo especificado.
- Comprobar que el método de clustering definido cumple con la formación de grupos de tamaño fijo.
Evaluar el algoritmo desarrollado utilizando conjuntos de datos de prueba y métricas.

## _**Dependencias**_
Librerías: networkx, matplotlib, scikit-learn, pymetis.

## _**Instalación**_
1. Instalar [Miniconda3](https://docs.anaconda.com/miniconda/) . Durante la instalación, marcar que se agregue conda al PATH.
2. Abrir la terminal en la carpeta del proyecto y ejecutar los siguientes comandos:
   ```sh
   pip install networkx
   pip install matplotlib
   pip install scikit-learn
   conda update conda
   conda install conda-forge::pymetis
    ```
## _**Referencias**_
- Zhang, F., Guo, H., Ouyang, D., Yang, S., Lin, X., & Tian, Z. (2024). Size-constrained community search on large networks: An effective and efficient solution. IEEE Transactions on Knowledge and Data Engineering, 36(1), 356-371. https://doi.org/10.1109/TKDE.2023.3280483