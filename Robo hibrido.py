from time import sleep
from robo_core import RoboHibrido, escrever_log


def main():
    worker = RoboHibrido()
    worker.start()
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        escrever_log("Recebido KeyboardInterrupt. Encerrando...")
        worker.stop()


if __name__ == "__main__":
    main()
