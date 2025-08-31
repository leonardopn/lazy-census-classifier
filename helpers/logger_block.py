def logger_block(title: str | None):
    """
    Cria um bloco de log para uma função específica.
    title: título do bloco de log.
    """

    if title:
        print(f"\n{'=' * 40} {title} {'=' * 40}\n")
    else:
        print(f"\n{'=' * 80}\n")
