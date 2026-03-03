"""
Management command to clean up duplicate model files.

Scans all metadata files in media/models/, groups models by their
data fingerprint (CSV content + target column), and removes duplicate
.pkl files — keeping only one copy per unique fingerprint.

Usage:
    python manage.py cleanup_duplicate_models          # dry run (shows what would be deleted)
    python manage.py cleanup_duplicate_models --apply  # actually delete duplicates
"""

import os
import json
import glob
import hashlib
from collections import defaultdict

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Remove duplicate model .pkl files that share the same data fingerprint'

    def add_arguments(self, parser):
        parser.add_argument(
            '--apply',
            action='store_true',
            help='Actually delete duplicates (default is dry-run)',
        )
        parser.add_argument(
            '--backfill',
            action='store_true',
            help='Compute and save fingerprints for metadata files that lack them',
        )

    def handle(self, *args, **options):
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        if not os.path.isdir(models_dir):
            self.stdout.write(self.style.WARNING('No models directory found.'))
            return

        apply_changes = options['apply']
        backfill = options['backfill']

        # ── Step 1: Backfill fingerprints for old metadata files ──
        if backfill:
            self._backfill_fingerprints(models_dir)

        # ── Step 2: Group metadata by fingerprint ──
        fingerprint_groups = defaultdict(list)  # fingerprint → [(meta_path, model_path, file_size)]
        no_fingerprint = []

        for meta_file in sorted(glob.glob(os.path.join(models_dir, '*_meta.json'))):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            fp = meta.get('data_fingerprint')
            model_path = meta.get('model_path', '')

            if not fp:
                no_fingerprint.append(meta_file)
                continue

            size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
            fingerprint_groups[fp].append({
                'meta_path': meta_file,
                'model_path': model_path,
                'size': size,
                'target': meta.get('target_col', '?'),
            })

        # ── Step 3: Report & clean ──
        total_saved = 0
        files_removed = 0

        for fp, entries in fingerprint_groups.items():
            # Deduplicate: find unique .pkl paths
            unique_pkl_paths = {}
            for e in entries:
                pkl = e['model_path']
                if pkl not in unique_pkl_paths:
                    unique_pkl_paths[pkl] = e

            if len(unique_pkl_paths) <= 1:
                continue

            # Keep the first (smallest dataset_id / oldest) .pkl, remove the rest
            keep_path = list(unique_pkl_paths.keys())[0]
            keep_entry = unique_pkl_paths[keep_path]

            self.stdout.write(self.style.SUCCESS(
                f'\n  Fingerprint: {fp[:16]}... | target="{keep_entry["target"]}"'
            ))
            self.stdout.write(f'    KEEP: {os.path.basename(keep_path)} ({keep_entry["size"]:,} bytes)')

            for pkl_path, entry in list(unique_pkl_paths.items())[1:]:
                self.stdout.write(self.style.WARNING(
                    f'    DELETE: {os.path.basename(pkl_path)} ({entry["size"]:,} bytes)'
                ))
                total_saved += entry['size']
                files_removed += 1

                if apply_changes:
                    # Update all metadata files pointing to this duplicate
                    for e2 in entries:
                        if e2['model_path'] == pkl_path:
                            try:
                                with open(e2['meta_path']) as f:
                                    m = json.load(f)
                                m['model_path'] = keep_path
                                with open(e2['meta_path'], 'w') as f:
                                    json.dump(m, f, indent=2)
                                self.stdout.write(f'      Updated {os.path.basename(e2["meta_path"])} → {os.path.basename(keep_path)}')
                            except Exception as ex:
                                self.stdout.write(self.style.ERROR(f'      Failed to update {e2["meta_path"]}: {ex}'))

                    # Delete the duplicate .pkl
                    try:
                        os.remove(pkl_path)
                        self.stdout.write(self.style.SUCCESS(f'      Deleted {pkl_path}'))
                    except OSError as ex:
                        self.stdout.write(self.style.ERROR(f'      Failed to delete: {ex}'))

        # ── Summary ──
        self.stdout.write('')
        if files_removed == 0:
            self.stdout.write(self.style.SUCCESS('No duplicate models found. Storage is clean!'))
        else:
            mb_saved = total_saved / (1024 * 1024)
            action = 'Freed' if apply_changes else 'Would free'
            self.stdout.write(self.style.SUCCESS(
                f'{action} {mb_saved:.1f} MB by removing {files_removed} duplicate model(s).'
            ))
            if not apply_changes:
                self.stdout.write(self.style.WARNING('  Run with --apply to actually delete files.'))

        if no_fingerprint:
            self.stdout.write(self.style.WARNING(
                f'\n  {len(no_fingerprint)} metadata file(s) have no fingerprint.'
                f'\n  Run with --backfill to compute and save fingerprints for them.'
            ))

    def _backfill_fingerprints(self, models_dir):
        """Compute fingerprints for old metadata files that don't have them."""
        count = 0
        for meta_file in sorted(glob.glob(os.path.join(models_dir, '*_meta.json'))):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if meta.get('data_fingerprint'):
                continue

            file_path = meta.get('file_path', '')
            target_col = meta.get('target_col', '')

            if not file_path or not os.path.exists(file_path) or not target_col:
                self.stdout.write(self.style.WARNING(
                    f'  Skip {os.path.basename(meta_file)}: missing file_path or target_col'
                ))
                continue

            # Compute fingerprint
            h = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            h.update(target_col.encode('utf-8'))
            fingerprint = h.hexdigest()

            meta['data_fingerprint'] = fingerprint
            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2)

            count += 1
            self.stdout.write(f'  Backfilled {os.path.basename(meta_file)} → {fingerprint[:16]}...')

        self.stdout.write(self.style.SUCCESS(f'  Backfilled {count} metadata file(s).\n'))
