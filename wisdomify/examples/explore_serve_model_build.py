from wisdomify.main.build_mar import build_archive_model


def main():
    build_archive_model(version=1, max_length=11,
                        checkpoint_file_name="wisdomifier.ckpt")


if __name__ == '__main__':
    main()