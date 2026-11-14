"""Điểm vào giao diện: ghép Model – View – Controller rồi chạy."""
from .controllers.main_controller import MainController
from .models.session import ImageSession
from .views.main_window import MainWindow


def create_app():
    model = ImageSession()
    view = MainWindow()
    return MainController(model, view)


def main():
    create_app().start()


if __name__ == "__main__":
    main()
