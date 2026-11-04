# Ejecución y arquitectura

Desde la raíz del proyecto en Windows, con pyodbc y un driver SQL Server instalados:

```powershell
$env:SQLSERVER_SERVER = 'lpc:.\AMBATO'
python -m src.db.auditoria_dw
python -m unittest discover -s tests -p 'test_auditoria_dw.py' -v
```

La instancia se verificó por servicios y consulta SELECT: AMBATO contiene DW y OLTP; BI solo OLTP. También admite --server y SQLSERVER_DATABASE/SQLSERVER_DRIVER. No hay contraseña: autenticación integrada. No se cambió requirements.txt ni se instalaron dependencias; pyodbc ya estaba disponible. La ejecución completa vuelve a consultar las bases y sustituye solo esta carpeta de auditoría: conservar una copia si se necesita contrastar snapshots posteriores. --metadata-only limita la consulta al catálogo. Cero ETL, modificaciones SQL o entrenamiento.

Capas futuras: conexión -> extracción por proceso -> auditoría -> agregación -> etiqueta -> características -> particiones con purga -> preprocesador ajustado en train -> modelos existentes -> evaluación -> visualización. Mantener datos/resultados separados por proceso y versión; no usar las constantes BOUNDS ni fórmulas de Ventas en otro proceso sin contrato. Procesos/clientes/proveedores se identifican por IDs operacionales, agregando versiones SCD por tiempo; no por nombre.

Los resultados incluyen únicamente metadatos y agregados, más ejemplos limitados de intervalos; no se vuelca la base. Definiciones de integración son evidencia textual obtenida con SELECT, nunca procedimientos ejecutados. La consistencia de las huellas es una comprobación acotada, no una garantía de que nadie más cambió SQL Server durante toda la auditoría.
