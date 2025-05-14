from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

options = webdriver.ChromeOptions()
options.add_argument("--ignore-certificate-errors")
options.add_argument("--ignore-ssl-errors")
options.add_argument("--disable-web-security")

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
driver.get("https://www.lfendesa.es/Inicio.aspx?tabid=9")

wait = WebDriverWait(driver, 20)
wait.until(EC.presence_of_element_located((By.ID, "_ctl0_temporadasDropDownList")))

temporada_select = Select(driver.find_element(By.ID, "_ctl0_temporadasDropDownList"))
temporadas = [opt.get_attribute("value") for opt in temporada_select.options if opt.get_attribute("value").isdigit()]

all_data = []
headers = []

def extract_data(temporada):
    global headers
    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("table", class_="tabla-estadistica")
    if not table:
        return []

    if not headers:
        headers = ["Temporada"] + [th.text.strip() for th in table.find_all("th")]

    rows = table.find_all("tr")[1:]
    season_data = []

    for row in rows:
        cols = [td.text.strip() for td in row.find_all("td")]
        if cols:
            season_data.append([temporada] + cols)

    return season_data

for temporada in temporadas:
    print(f"Extrayendo temporada {temporada}...")
    temporada_select = Select(driver.find_element(By.ID, "_ctl0_temporadasDropDownList"))
    temporada_select.select_by_value(temporada)
    time.sleep(3)

    # Obtener número total de páginas
    try:
        wait.until(EC.presence_of_element_located((By.ID, "_ctl0_rankingsAcumDataPager__ctl0_Label1")))
        total_pages_elem = driver.find_element(By.ID, "_ctl0_rankingsAcumDataPager__ctl0_Label1")
        total_pages = int(total_pages_elem.text.strip())
    except Exception:
        total_pages = 1

    for page in range(1, total_pages + 1):
        print(f"  Página {page}/{total_pages}")
        if page > 1:
            driver.execute_script(f"__doPostBack('_ctl0$rankingsAcumDataPager$_ctl2$_ctl{page}','')")
            time.sleep(2)
        datos = extract_data(temporada)
        all_data.extend(datos)

driver.quit()

df = pd.DataFrame(all_data, columns=headers)
df.to_csv("lfendesa_rankings_temporadas.csv", index=False, encoding="utf-8")
print("✅ Datos guardados en lfendesa_rankings_temporadas.csv")
