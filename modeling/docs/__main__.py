import os
import time
import click
import requests
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from modeling.config import settings, logger
from modeling.config.settings import LogLevel

class SphinxFileChangeHandler(FileSystemEventHandler):
    def __init__(self, src_dir, build_dir):
        self.src_dir = src_dir
        self.build_dir = build_dir
        self.server_process = None

    def build_sphinx(self):
        logger.info("Rebuilding Sphinx documentation...")
        subprocess.run(["sphinx-build", "-b", "html", self.src_dir, self.build_dir])
        logger.info("Sphinx documentation rebuilt.")

    def on_modified(self, event):
        if os.path.dirname(event.src_path).startswith(self.src_dir) and event.src_path.endswith((".rst", "conf.py", ".puml")):
            logger.info(event)
            self.build_sphinx()


class HttpServer:
    def __init__(self, build_dir):
        self.build_dir = build_dir
        self.http_server_args = [
            "python",
            "-m",
            "http.server",
            "--directory",
            self.build_dir,
        ]
        self.server_process = None

    def start_server(self):
        self.server_process = subprocess.Popen(
            self.http_server_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("Http server started.")

    def stop_server(self):
        if self.server_process:
            self.server_process.kill()
            logger.info("Http server stopped.")

    def restart_server(self):
        self.stop_server()
        time.sleep(1)  # Ensure the server process is fully terminated
        self.start_server()


class DocumentationBuilder(SphinxFileChangeHandler, HttpServer):
    def __init__(self, src_dir, build_dir, cache_dir):
        SphinxFileChangeHandler.__init__(self, src_dir, build_dir)
        HttpServer.__init__(self, build_dir)
        self.dependencies_dir = cache_dir
        self.plantuml_jar_file = settings.DOCS.PLANTUML_JAR

    def build(self):
        self.build_sphinx()
        return self

    def serve(self):
        self.start_server()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_server()

    def watch(self):
        observer = Observer()
        observer.schedule(self, path=self.src_dir, recursive=True)
        observer.start()
        logger.info("Watching for file changes...")
        self.start_server()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_server()
            observer.stop()
            observer.join()

    def on_modified(self, event):
        super().on_modified(event)
        if event.is_directory and event.src_path.endswith(self.build_dir):
            self.restart_server()

    def download_dependencies(self):
        logger.debug("Running download script !!!!")
        logger.warning(self.dependencies_dir)
        filename = os.path.join(self.dependencies_dir, os.path.basename(self.plantuml_jar_file))
        logger.warning(filename)
        if not os.path.exists(filename):
            os.makedirs(self.dependencies_dir, exist_ok=True)
            response = requests.get(self.plantuml_jar_file)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                logger.info(f"File downloaded: {filename}")
            else:
                logger.error(f"Failed to download file: {response.status_code}")


@click.group("docs")
def cli():
    """Documentation builder CLI tool"""
    pass

@cli.command()
@click.option('--debug/--no-debug', '-d/-D', default=False, help='Enable debug mode')
def build(debug):
    """Build Sphinx documentation"""
    if debug:
        settings.LOG_LEVEL = LogLevel.DEBUG
    
    SCR_DIR = os.path.abspath(settings.DOCS.SOURCE_DIR)
    BLD_DIR = os.path.abspath(settings.DOCS.BUILD_DIR)
    CACHE_DIR = os.path.abspath(settings.DOCS.CACHE_DIR)
    
    logger.debug(f"Source dir: {SCR_DIR}")
    logger.debug(f"Build dir: {BLD_DIR}")
    logger.debug(f"Cache dir: {CACHE_DIR}")
    
    doc = DocumentationBuilder(src_dir=SCR_DIR, build_dir=BLD_DIR, cache_dir=CACHE_DIR)
    doc.download_dependencies()
    doc.build()

@cli.command()
@click.option('--debug/--no-debug', '-d/-D', default=False, help='Enable debug mode')
def watch(debug):
    """Watch for changes and rebuild documentation"""
    if debug:
        settings.LOG_LEVEL = LogLevel.DEBUG
    
    SCR_DIR = os.path.abspath(settings.DOCS.SOURCE_DIR)
    BLD_DIR = os.path.abspath(settings.DOCS.BUILD_DIR)
    CACHE_DIR = os.path.abspath(settings.DOCS.CACHE_DIR)
    
    doc = DocumentationBuilder(src_dir=SCR_DIR, build_dir=BLD_DIR, cache_dir=CACHE_DIR)
    doc.download_dependencies()
    doc.build().watch()

@cli.command()
@click.option('--debug/--no-debug', '-d/-D', default=False, help='Enable debug mode')
def serve(debug):
    """Start the documentation server"""
    if debug:
        settings.LOG_LEVEL = LogLevel.DEBUG
    
    SCR_DIR = os.path.abspath(settings.DOCS.SOURCE_DIR)
    BLD_DIR = os.path.abspath(settings.DOCS.BUILD_DIR)
    CACHE_DIR = os.path.abspath(settings.DOCS.CACHE_DIR)
    
    doc = DocumentationBuilder(src_dir=SCR_DIR, build_dir=BLD_DIR, cache_dir=CACHE_DIR)
    doc.download_dependencies()
    doc.build().serve()

if __name__ == '__main__':
    cli()
