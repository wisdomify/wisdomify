from wisdomify.main.build import build_archive_model


def main():
    build_archive_model(version=0, max_length=11,
                        checkpoint_file_name="wisdomify_def_epoch=19_train_loss=0.00.ckpt")


if __name__ == '__main__':
    main()
