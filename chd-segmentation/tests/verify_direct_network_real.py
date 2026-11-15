import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import os
from pathlib import Path
import ssl
import subprocess
import sys
from threading import Thread
from time import perf_counter
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT/'tests'))

from verify_delivery_real import MetadataWorkspace
from chd_segmentation.delivery.assets import download_file, git_command


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def verify_download(root, openssl):
    source = PROJECT/'vendor/MedSAM2/LICENSE'
    body = source.read_bytes()
    (root/'LICENSE').write_bytes(body)
    cert, key = root/'certificate.pem', root/'key.pem'
    subprocess.run([openssl, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(cert), '-days', '1',
                    '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1'],
                   check=True, capture_output=True)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(root)))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    previous = os.environ.get('SSL_CERT_FILE')
    os.environ['SSL_CERT_FILE'] = str(cert)
    started = perf_counter()
    try:
        target = root/'downloaded-LICENSE'
        url = f'https://127.0.0.1:{server.server_port}/LICENSE'
        result = download_file(url, target, len(body), hashlib.sha256(body).hexdigest())
        assert target.read_bytes() == body
        stamp = target.stat().st_mtime_ns
        assert download_file(url, target, len(body), hashlib.sha256(body).hexdigest())['reused']
        assert target.stat().st_mtime_ns == stamp
        with unittest.TestCase().assertRaises(ValueError):
            download_file(url, root/'wrong-digest', len(body), '0'*64)
        assert not (root/'wrong-digest.pending').exists()
        return dict(kind='HTTPS_download', seconds=perf_counter()-started, bytes=len(body),
                    downloaded=result, digest_mismatch_rejected=True, existing_file_preserved=True)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()
        if previous is None:
            os.environ.pop('SSL_CERT_FILE', None)
        else:
            os.environ['SSL_CERT_FILE'] = previous


def verify_git(root, poison):
    repo = root/'repository'
    repo.mkdir()
    clean = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    clean.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
    subprocess.run(['git', '-C', str(repo), 'init', '--template='], env=clean,
                   check=True, capture_output=True)
    keys = ['http.proxy', 'http.https://github.com/.proxy', 'remote.origin.proxy']
    for name in keys:
        subprocess.run(['git', '-C', str(repo), 'config', '--local', name, poison],
                       env=clean, check=True, capture_output=True)
    before = (repo/'.git/config').read_bytes()
    effective = git_command(repo, ['config', '--get-urlmatch', 'http.proxy',
                                  'https://github.com/bowang-lab/MedSAM2.git']).stdout.strip()
    assert effective == '', f'旧Git代理仍然生效：{effective}'
    explicit = 'http://127.0.0.1:7897'
    selected = git_command(repo, ['config', '--get-urlmatch', 'http.proxy',
                                 'https://github.com/bowang-lab/MedSAM2.git'], explicit).stdout.strip()
    assert selected == explicit
    assert git_command(repo, ['config', '--get', 'remote.origin.proxy']).stdout.strip() == ''
    assert (repo/'.git/config').read_bytes() == before
    return dict(kind='Git_effective_configuration', direct_proxy='',
                explicit_proxy_preserved=True, local_configuration_unchanged=True)


def verify_dependencies(root, poison):
    from chd_segmentation.delivery.network import subprocess_environment
    before = dict(os.environ)
    env = subprocess_environment()
    result = subprocess.run([sys.executable, '-B', '-m', 'pip', 'config', 'debug'],
                            env=env, check=True, capture_output=True, text=True)
    assert poison not in result.stdout
    assert env['PIP_CONFIG_FILE'] == os.devnull
    assert env['NO_PROXY'] == '*'
    assert not any(k.upper() in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'PIP_PROXY') for k in env)
    explicit = subprocess_environment('http://127.0.0.1:7897')
    assert explicit['HTTPS_PROXY'] == 'http://127.0.0.1:7897'
    assert explicit['NO_PROXY'] == ''
    assert dict(os.environ) == before
    return dict(kind='actual_pip_configuration', inherited_proxy_absent=True,
                pip_configuration_disabled=True, environment_unchanged=True,
                full_dependency_install_performed=False)


def run(args):
    sys.path.insert(0, args.simulator)
    from common import clean_owned, files_under
    with MetadataWorkspace(args.work_root, args.report, clean_owned, files_under) as workspace:
        poison = 'http://127.0.0.1:1'
        pip_config = workspace.root/'pip.ini'
        pip_config.write_text('[global]\nproxy = '+poison+'\n', encoding='utf-8')
        previous = dict(os.environ)
        os.environ.update(HTTP_PROXY=poison, HTTPS_PROXY=poison, ALL_PROXY=poison,
                          NO_PROXY='', PIP_PROXY=poison, PIP_CONFIG_FILE=str(pip_config),
                          GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='http.proxy',
                          GIT_CONFIG_VALUE_0=poison)
        poisoned = dict(os.environ)
        try:
            checks = [('download', lambda: verify_download(workspace.root, args.openssl)),
                      ('git', lambda: verify_git(workspace.root, poison)),
                      ('dependencies', lambda: verify_dependencies(workspace.root, poison))]
            for name, check in checks:
                if args.only in ('all', name):
                    result = check()
                    assert dict(os.environ) == poisoned
                    workspace.results.append(dict(result, ct_voxels_read=0, optimizer_updates=0))
        finally:
            os.environ.clear()
            os.environ.update(previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-root', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--only', choices=('all', 'download', 'git', 'dependencies'), default='all')
    parser.add_argument('--simulator', default='D:/projects/test')
    parser.add_argument('--openssl', default='C:/msys64/usr/bin/openssl.exe')
    run(parser.parse_args())
