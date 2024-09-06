from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

def hello_airflow():
    print("Hello, Airflow!")

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

dag = DAG(
    'simple_dag',
    default_args=default_args,
    description='A simple DAG to test Airflow',
    schedule_interval='@once',
    start_date=days_ago(1),
)

task = PythonOperator(
    task_id='hello_task',
    python_callable=hello_airflow,
    dag=dag,
)