"""Evict page cache for files we own under given roots (posix_fadvise DONTNEED). No root needed."""
import os, sys
n = 0; b = 0
for root in sys.argv[1:]:
    for dp, _, fs in os.walk(os.path.expanduser(root)):
        for f in fs:
            p = os.path.join(dp, f)
            try:
                if os.path.islink(p) or os.stat(p).st_uid != os.getuid():
                    continue
                fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
                n += 1; b += os.path.getsize(p)
            except OSError:
                pass
print(f"evicted {n} files, {b/1e9:.1f} GB")
