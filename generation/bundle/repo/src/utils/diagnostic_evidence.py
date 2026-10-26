"""Opt-in durable evidence. Failure to preserve evidence aborts the diagnostic."""
import json
import os


class EvidenceWriteError(RuntimeError):
    pass


class JsonlEvidence:
    def __init__(self, path):
        self.file = open(path, 'x', encoding='utf-8')
        self.sequence = 0

    def __call__(self, event):
        try:
            self.file.write(json.dumps({'sequence': self.sequence, **event}, ensure_ascii=False) + '\n')
            self.file.flush()
            os.fsync(self.file.fileno())
            self.sequence += 1
        except Exception as exc:
            raise EvidenceWriteError('Diagnostic evidence could not be persisted') from exc

    def close(self):
        self.file.close()
