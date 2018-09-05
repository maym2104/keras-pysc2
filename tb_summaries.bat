@echo off
set counter=6000
REM dir out\summary\
for /D %%G in (out\summary\*) do (
	call :loop_body "%%G"
 )
GOTO :eof

:loop_body
  set /A counter+=1
  START "tensorboard %1 on %counter%" tensorboard --logdir %1\train --port %counter%
  GOTO :eof