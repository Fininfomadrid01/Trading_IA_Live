# 🛰️ MANUAL DE SINCRONIZACIÓN: LABORATORIO -> GITHUB

Para que tu Dashboard de Trading refleje los nuevos resultados globales (NASDAQ, FTSE, S&P) y utilice los parámetros optimizados por la IA, debes sincronizar los siguientes archivos.

## 1. Archivos Críticos a Sincronizar
Debes subir/actualizar estos archivos en tu repositorio de GitHub:

1.  **Dashboard Global:** `github_automation/generador_reporte.py`
2.  **Lanzador Maestro:** `Codigo_10_Lanzador_IBKR.py`
3.  **Memoria Final:** `Documentacion_TFM/TFM_Documento_Completo.md`

## 2. Instrucciones Paso a Paso (Vía Web)

1.  Entra en tu repositorio de **GitHub** desde el navegador.
2.  **Para el Reporte:**
    *   Navega hasta la carpeta `github_automation`.
    *   Haz clic en **Add file** -> **Upload files**.
    *   Arrastra el archivo `generador_reporte.py` de tu ordenador.
3.  **Para el Lanzador y la Memoria:**
    *   Vuelve a la raíz del repositorio.
    *   Sube los archivos correspondientes repitiendo el proceso de **Upload files**.
4.  **Confirmar:** Escribe un mensaje de commit como "Actualización Global Optimizada" y pulsa **Commit changes**.

## 3. Verificación de la Nube
Una vez subidos:
1.  Ve a la pestaña **Actions** en GitHub.
2.  Si tienes una tarea programada, puedes lanzarla manualmente seleccionando el workflow y pulsando **Run workflow**.
3.  Comprueba que el `README.md` de tu repositorio ahora muestra las banderas de 🇺🇸, 🇬🇧 y 🇪🇸 y el estado de los tres mercados.

---
**Nota:** Si usas la consola de comandos, simplemente ejecuta:
`git add .`
`git commit -m "Upgrade Global Trading IA"`
`git push`
