import os
import pkg_resources

def calc_package_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# 크기 제한 설정 (KB 단위)
SIZE_LIMIT = 50000  # 5MB

# 작은 패키지만 저장
with open('requirements.txt', 'w') as f:
    for dist in pkg_resources.working_set:
        try:
            path = os.path.join(dist.location, dist.project_name)
            size = calc_package_size(path) / 1000  # KB로 변환
            
            if size < SIZE_LIMIT:
                f.write(f"{dist.project_name}=={dist.version}\n")
        except OSError:
            continue