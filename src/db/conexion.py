"""Configuración sin secretos persistidos y frontera SELECT única para la auditoría."""
from dataclasses import dataclass
import os
import re


DATABASES = {'WideWorldImportersDW', 'WideWorldImporters'}


def identificador(value):
    if not isinstance(value, str) or not value or len(value)>128 or '\x00' in value:
        raise ValueError('Identificador SQL inválido')
    return '['+value.replace(']',']]')+']'


def validar_select(sql):
    """Acepta un único SELECT; rechaza comentarios, lotes y SELECT INTO.

    No pretende interpretar T-SQL general: es una frontera deliberadamente
    restrictiva para las plantillas fijas de este módulo. Identificadores y
    literales se eliminan antes de revisar palabras reservadas.
    """
    if not isinstance(sql,str) or not sql.strip(): raise ValueError('SELECT requerido')
    token=re.compile(r"N?'(?:''|[^'])*'|\[(?:\]\]|[^\]])*\]")
    clean=token.sub(' ',sql)
    if any(x in clean for x in [';','--','/*','*/']): raise ValueError('Lotes/comentarios no autorizados')
    if not re.match(r'^\s*SELECT\b',clean,re.I): raise ValueError('Solo SELECT')
    if re.search(r'\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|INTO|GRANT|DENY|REVOKE|BACKUP|RESTORE|DBCC|OPENROWSET|OPENQUERY|BULK|NEXT|SET|USE)\b',clean,re.I):
        raise ValueError('Operación fuera del alcance de lectura')
    return sql


@dataclass(frozen=True)
class Configuracion:
    server: str
    database: str = 'WideWorldImportersDW'
    driver: str = 'ODBC Driver 17 for SQL Server'
    timeout: int = 30

    @classmethod
    def entorno(cls,server=None,database=None):
        host=server or os.environ.get('SQLSERVER_SERVER')
        if not host: raise ValueError('Falta SQLSERVER_SERVER o --server; verificar la instancia local primero')
        return cls(host,database or os.environ.get('SQLSERVER_DATABASE','WideWorldImportersDW'),os.environ.get('SQLSERVER_DRIVER','ODBC Driver 17 for SQL Server'))

    def conectar(self):
        import pyodbc
        if self.database not in DATABASES:raise ValueError('Base fuera del alcance autorizado')
        if self.driver not in pyodbc.drivers():raise ValueError('Driver ODBC configurado no disponible')
        if any(c in self.server for c in ';{}\x00') or not self.server:raise ValueError('Servidor inválido')
        if not 1<=self.timeout<=120:raise ValueError('Timeout inválido')
        # Sin credenciales SQL: autenticación integrada del usuario Windows.
        # readonly/ApplicationIntent son indicaciones al driver, no una barrera
        # de permisos: la protección efectiva aquí es no emitir escrituras.
        return pyodbc.connect(f'DRIVER={{{self.driver}}};SERVER={self.server};DATABASE={self.database};Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=yes;ApplicationIntent=ReadOnly',
                              timeout=self.timeout,autocommit=True,readonly=True)


class Lector:
    def __init__(self,connection):
        self.connection=connection;self.consultas=[]
        self.connection.timeout=60

    def select(self,sql,params=()):
        validar_select(sql)
        cursor=self.connection.cursor()
        record={'sql':sql,'parameters':list(params),'operation':'SELECT','status':'started'}
        self.consultas.append(record)
        try:
            cursor.execute(sql,*params)
            names=[c[0] for c in cursor.description]
            rows=[dict(zip(names,row)) for row in cursor.fetchall()]
            record.update(rows_returned=len(rows),status='successful')
            return rows
        except Exception:
            record['status']='failed'
            raise
        finally:cursor.close()

    def close(self):self.connection.close()
