py manage.py runserver
timeout /t 1
start chrome
timeout /t 5
start chrome "http://127.0.0.1:8000/app/"