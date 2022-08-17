import cv2


class VideoRecord:
    def __init__(self, filename, fps=30, is_rgb=False) -> None:
        self.filename = filename
        self.writer = None
        self.fps = fps
        self.is_rgb = is_rgb

    def update(self, img):
        if self.filename is not None:
            if self.writer is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (w, h))

            if self.is_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.writer.write(img)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print("End record")
