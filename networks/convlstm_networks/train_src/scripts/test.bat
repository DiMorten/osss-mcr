@echo off
set dataset=%1
set source=%2
echo %dataset%_%source%
echo "%dataset%_%source%2"
set a='abc'
set b="cde"
echo %a% and %b%
cd ..
ls
python main.py