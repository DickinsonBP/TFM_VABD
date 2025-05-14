from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager

from bs4 import BeautifulSoup

import pandas as pd
import time

# Configurar el WebDriver usando webdriver_manager para manejar la instalación del ChromeDriver
# Añadir opciones para ingorar errores SSL
options = Options()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_argument('--disable-web-security')
service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# URL de la página de resultados de partidos
url = "https://www.lfendesa.es/Inicio.aspx?tabid=9"
driver.get(url)

# Esperar hasta que la tabla de resultados esté presente
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'tabla-estadistica')))

all_data = []

def extract_data():
    # Obtener el contenido HTML de la página actual
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Encontrar la tabla de resultados de partidos
    results_table = soup.find('table', {'class': 'tabla-estadistica'})
    
    if results_table:
        headers = [header.text for header in results_table.find_all('th')]
        rows = results_table.find_all('tr')[1:]  # Omitir la fila de encabezado
        table_data = []
        
        for row in rows:
            columns = row.find_all('td')
            row_data = [column.text.strip() for column in columns]
            table_data.append(row_data)
        
        # Agregar los datos de la página actual a la lista total
        all_data.extend(table_data)
        return headers
    return []

# Extraer datos de la primera página
headers = extract_data()

page_number = 2
while True:
    try:
        # Ejecutar el script __doPostBack para ir a la siguiente página
        driver.execute_script(f"__doPostBack('_ctl0$rankingsAcumDataPager$_ctl2$_ctl{page_number}','')")
        
        # Esperar un momento para que la página se cargue
        time.sleep(2)
        
        # Extraer datos de la nueva página
        new_headers = extract_data()
        if not new_headers:
            break  # Si no se encuentran nuevos datos, salir del bucle
        
        page_number += 1
    except:
        break

# Cerrar el navegador
driver.quit()

# Crear un DataFrame con todos los datos recopilados
df = pd.DataFrame(all_data, columns=headers)

# Mostrar los primeros registros
print(df.head())

# Guardar los datos en un archivo CSV
output_file = 'lfendesa_resultados.csv'
df.to_csv(output_file, index=False)
print(f'Datos guardados en {output_file}')
