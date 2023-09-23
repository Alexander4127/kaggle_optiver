# Optiver - Trading at the Close

[Ссылка на Kaggle.](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview)

## Что тут лежит

1. В `example_test_files` лежат примеры тестовых сетов, которые нам могут дать.
2. В `optiver2023` по сути находится файл `public_timeseries_testing_util.py` только с названием competition.
Там лежит странная для линукса, поэтому на винде ничего не запускается. Поэтому я создал рядом простой питонячий файл, но
пока что не понял, что нужно указывать в поле `self.group_id_column`.
3. Также положил сюда нотбук `subm.ipynb`, который можно засылать в качестве константного прогноза.

## Как вообще устроены посылки

Нам нужно присылать кэгловский нотбук ([вот мой](https://www.kaggle.com/code/alexander4127/notebook244976138f)), который надо для этого заранее сохранить там и прогнать (это делается автоматически).
Кроме того, перед сохранением надо выключить подключение к интернету у нотбука - в правой панельке проматываем вниз и в `Notebook options` выключаем инет.

В нотбуке показано, как использовать функцию `make_env()`, а затем как пользоваться полученной API через её метод `iter_test()`, который бегает по тестовому множеству.
Этот метод составляет файл с ответами модели за нас, нам нужно только класть правильный таргет. На данном этапе он пишет, что написан не оптимально, хз, надо ли этот файлик переписывать, но зашло и так.


