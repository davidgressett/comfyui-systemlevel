import os
import torch
import csv
import itertools
import hashlib
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import time

class CartesianCSVNode:
    combined_rows = []
    last_row = 0
    last_input = None
    file_timestamps = {}
    reset_flag = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_files": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Enter file paths, one per line (comma-separated = concatenate).",
                        "lines": 10
                    }
                ),
                "reset": ("BOOLEAN", {"default": False}),
                "row_index": ("INT", {"default": -1, "min": -1}),
                "show_combined_text": ("BOOLEAN", {"default": False, "label": "Show Combined Rows Text"}),
            }
        }

    RETURN_TYPES = (
        # 5 images
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        # 10 text outputs
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING", "STRING",
        # 10 string outputs
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING", "STRING",
        # 5 integers
        "INT", "INT", "INT", "INT", "INT",
        # 5 floats
        "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT",
        # 5 additional metadata
        "INT",     # row_count
        "STRING",  # combined_rows_text
        "BOOLEAN", # no_more_rows
        "INT",     # last_row
        "INT",     # next_row
    )

    RETURN_NAMES = (
        "Image 0", "Image 1", "Image 2", "Image 3", "Image 4",
        "Text 0", "Text 1", "Text 2", "Text 3", "Text 4",
        "Text 5", "Text 6", "Text 7", "Text 8", "Text 9",
        "String 0", "String 1", "String 2", "String 3", "String 4",
        "String 5", "String 6", "String 7", "String 8", "String 9",
        "Number 0", "Number 1", "Number 2", "Number 3", "Number 4",
        "Float 0", "Float 1", "Float 2", "Float 3", "Float 4",
        "Row Count", "Combined Rows", "No More", "Last Row", "Next Row"
    )

    # Corrected: we need 40 booleans here to match RETURN_TYPES length if our framework uses them.
    OUTPUT_IS_LIST = (
        False, False, False, False, False,  # 5 images
        False, False, False, False, False,  # 5 text
        False, False, False, False, False,  # 5 text (again)
        False, False, False, False, False,  # 5 string
        False, False, False, False, False,  # 5 string
        False, False, False, False, False,  # 5 number
        False, False, False, False, False,  # 5 float
        False, False, False, False, False   # row_count, combined_rows, no_more, last_row, next_row
    )

    CATEGORY = "SystemLevel/Complex Input"
    FUNCTION = "execute"

    def execute(self, csv_files, reset, row_index, show_combined_text):
        """
        - Each line in 'csv_files' is a group (so multiple lines => cartesian product).
        - If a single line has multiple CSV paths separated by commas, we concatenate them,
          provided columns are the same.
        """

        # Parse multiline input -> list of groups (each group can have 1 or more CSV files)
        groups = self._parse_input(csv_files)

        if reset and self.reset_flag:
            self.last_row = 0

        # Reset conditions: new input, reset flag, or file timestamps changed
        if ((reset and not self.reset_flag)
            or self._input_changed(groups)
            or self._timestamps_changed(groups)):
            self.combined_rows = self._validate_and_combine(groups)
            self.file_timestamps = self._get_file_timestamps(groups)
            self.last_row = 0
            self.last_input = groups
            self.reset_flag = True

        if not reset:
            self.reset_flag = False

        # Process the current or selected row
        if row_index >= 0:
            self.last_row = row_index

        if not self.combined_rows:
            # Edge case: no rows at all, return placeholders
            # Just return 40 placeholders (images, strings, etc.)
            return self._empty_outputs()

        last_row = self.last_row
        row = self.combined_rows[self.last_row]

        no_more_rows = False
        if (self.last_row + 1) >= len(self.combined_rows):
            no_more_rows = True

        self.last_row = (self.last_row + 1) % len(self.combined_rows)
        next_row = self.last_row

        # Process the row data
        outputs = self._process_row(row)

        # Prepare the final return
        images = [outputs.get(f"image_{i}", self._load_placeholder_image()) for i in range(5)]
        texts = [outputs.get(f"text_{i}", "") for i in range(10)]
        strings = [outputs.get(f"string_{i}", "") for i in range(10)]
        numbers = [outputs.get(f"number_{i}", 0) for i in range(5)]
        floats = [outputs.get(f"float_{i}", 0.0) for i in range(5)]

        combined_rows_text = ""
        if show_combined_text:
            combined_rows_text = self._format_combined_rows(self.combined_rows)
        row_count = len(self.combined_rows)

        return (
            images[0], images[1], images[2], images[3], images[4],
            texts[0], texts[1], texts[2], texts[3], texts[4],
            texts[5], texts[6], texts[7], texts[8], texts[9],
            strings[0], strings[1], strings[2], strings[3], strings[4],
            strings[5], strings[6], strings[7], strings[8], strings[9],
            numbers[0], numbers[1], numbers[2], numbers[3], numbers[4],
            floats[0], floats[1], floats[2], floats[3], floats[4],
            row_count,
            combined_rows_text,
            no_more_rows,
            last_row,
            next_row
        )

    def _parse_input(self, csv_files_str):
        """
        Returns a list of "groups". Each "group" is a list of CSV file paths.
        If a single line has multiple comma-separated paths, they are concatenated.
        """
        lines = csv_files_str.splitlines()
        groups = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # If you want to concatenate CSVs in a line, split by comma:
            paths = [p.strip() for p in line.split(",") if p.strip()]
            groups.append(paths)
        return groups

    def _load_placeholder_image(self):
        """
        Return a placeholder image of shape [3, 512, 512].
        """
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        img_array = np.array(img).astype(np.float32) / 255.0
        # Convert from [H, W, 3] -> [3, H, W]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return img_tensor

    def _empty_outputs(self):
        """
        Return an all-placeholder tuple if the combined_rows list is empty.
        """
        placeholder_img = self._load_placeholder_image()
        empty_string = ""
        zero_int = 0
        zero_float = 0.0

        # 5 images
        out = [placeholder_img]*5
        # 10 text
        out.extend([empty_string]*10)
        # 10 string
        out.extend([empty_string]*10)
        # 5 numbers
        out.extend([zero_int]*5)
        # 5 floats
        out.extend([zero_float]*5)
        # row_count, combined_text, no_more_rows, last_row, next_row
        out.extend([0, "", True, 0, 0])
        return tuple(out)

    def _timestamps_changed(self, groups):
        """Check if the modification timestamps of any files have changed."""
        current_timestamps = self._get_file_timestamps(groups)
        return self.file_timestamps != current_timestamps

    def _get_file_timestamps(self, groups):
        """
        Flatten all CSV file paths in groups and get their timestamps.
        Returns a dict {path: mtime or None}.
        """
        timestamps = {}
        for group in groups:
            for path in group:
                if os.path.exists(path):
                    timestamps[path] = os.path.getmtime(path)
                else:
                    timestamps[path] = None
        return timestamps

    def _input_changed(self, groups):
        """Check if the input file list (including concatenation groups) has changed."""
        return self.last_input != groups

    @staticmethod
    def _validate_and_combine(groups):
        """
        - For each group (which may contain multiple CSVs for concatenation),
          load and verify columns match, then combine (concatenate) into a single spreadsheet.
        - Then produce cartesian product across these spreadsheets if there's > 1 group.
        """
        # 1) Load/concatenate each group into a single 'spreadsheet'
        spreadsheets = []
        for group in groups:
            # For each group with multiple paths -> concatenate
            combined_rows = CartesianCSVNode._load_and_concatenate_csvs(group)
            spreadsheets.append(combined_rows)

        # 2) If more than one group => cartesian product across them
        if len(spreadsheets) > 1:
            cartesian_rows = []
            for combo in itertools.product(*spreadsheets):
                combined_row = {}
                for row_dict in combo:
                    combined_row.update(row_dict)
                cartesian_rows.append(combined_row)
            return cartesian_rows

        # 3) Only one group => single spreadsheet
        return spreadsheets[0] if spreadsheets else []

    @staticmethod
    def _load_and_concatenate_csvs(paths):
        """
        Concatenate multiple CSV files (same columns) into one list of dicts.
        Raise error if columns mismatch.
        """
        all_rows = []
        expected_columns = None
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV not found: {path}")
            with open(path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                # Check columns
                if rows:
                    these_columns = rows[0].keys()
                    if expected_columns is None:
                        # First CSV sets the column list
                        expected_columns = set(these_columns)
                    else:
                        # Must have the same columns
                        if set(these_columns) != expected_columns:
                            raise ValueError(
                                f"Mismatched columns in {path}. "
                                f"Expected {sorted(expected_columns)}, got {sorted(these_columns)}"
                            )
                all_rows.extend(rows)
        return all_rows

    def _process_row(self, row):
        """Parse out I_x, T_x, S_x, N_x, F_x columns from the row, then load or convert them."""
        data = self._parse_columns(row)
        outputs = {}

        # Load images (up to 5)
        for i in range(5):
            img_path = data["I"][i]
            if img_path:
                outputs[f"image_{i}"] = self._load_image(img_path)

        # 10 text fields
        for i in range(10):
            if data["T"][i]:
                text_data = self._load_text(data["T"][i])
                outputs[f"text_{i}"] = text_data if text_data is not None else ""

        # 10 string fields
        for i in range(10):
            if data["S"][i]:
                outputs[f"string_{i}"] = data["S"][i]

        # 5 integer fields
        for i in range(5):
            if data["N"][i]:
                try:
                    outputs[f"number_{i}"] = int(data["N"][i])
                except ValueError:
                    outputs[f"number_{i}"] = 0

        # 5 float fields
        for i in range(5):
            if data["F"][i]:
                try:
                    outputs[f"float_{i}"] = float(data["F"][i])
                except ValueError:
                    outputs[f"float_{i}"] = 0.0

        return outputs

    @staticmethod
    def _load_image(image_path):
        """Load an image from a file path, keep its size, and return a float32 PyTorch tensor [3, H, W]."""
        try:
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)  # Handle EXIF orientation
            img = img.convert("RGB")

            # Convert to NumPy [H,W,3], then to float32 normalized [0..1]
            img_array = np.array(img).astype(np.float32) / 255.0

            # Reorder to [3,H,W]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            return img_tensor
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")
            # Return a placeholder shape [3,512,512]
            return torch.zeros((3, 512, 512), dtype=torch.float32)

    @staticmethod
    def _load_text(file_path):
        """Return the file's entire text as a string (UTF-8)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file from {file_path}: {e}")
            return None

    @staticmethod
    def _parse_columns(row):
        """
        Collect up to:
        I_0..I_4, T_0..T_9, S_0..S_9, N_0..N_4, F_0..F_4
        """
        parsed_data = {
            "I": [None] * 5,  # images
            "T": [None] * 10, # text
            "S": [None] * 10, # strings
            "N": [None] * 5,  # integers
            "F": [None] * 5   # floats
        }
        for key, value in row.items():
            if "_" not in key:
                continue
            parts = key.split("_", 2)
            if len(parts) < 2:
                continue
            type_prefix, idx = parts[0], parts[1]
            if idx.isdigit():
                index = int(idx)
                if type_prefix in ["T", "S"] and 0 <= index < 10:
                    parsed_data[type_prefix][index] = value
                elif type_prefix in ["I", "N", "F"] and 0 <= index < 5:
                    parsed_data[type_prefix][index] = value
        return parsed_data

    def _format_combined_rows(self, rows):
        """Format combined rows into a string for debugging/logging."""
        if not rows:
            return "No combined rows available."
        lines = []
        for i, row in enumerate(rows):
            row_str = [f"  {k}: {v}" for k, v in row.items()]
            lines.append(f"Row {i+1}:\n" + "\n".join(row_str))
        return "\n\n".join(lines)

    @classmethod
    def IS_CHANGED(cls, csv_files, reset, row_index, show_combined_text):
        """
        Called by your automation to see if the node needs re-execution.
        - If row_index is -1, always re-run.
        - If reset is True, re-run.
        - Otherwise, hash the string inputs.
        """
        if row_index == -1:
            return float('nan')  # Forces re-execution
        if reset:
            return float('nan')

        input_str = f"{csv_files}-{row_index}-{show_combined_text}"
        return hashlib.sha256(input_str.encode('utf-8')).hexdigest()
