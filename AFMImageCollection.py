from AFMImage import AFMImage
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import visualizations as vis
from datetime import datetime

class AFMImageCollection:
    def __init__(self, folder_path, start_datetime=None, end_datetime=None):
        """
        Initialize AFMImageCollection with optional datetime filtering.
        
        Parameters:
        -----------
        folder_path : str
            Path to folder containing .ibw files
        start_datetime : datetime, optional
            Only include images taken at or after this datetime
        end_datetime : datetime, optional
            Only include images taken at or before this datetime
        """
        self.images = self.load_ibw_files_from_folder(folder_path, start_datetime, end_datetime)

    def load_ibw_files_from_folder(self, folder_path, start_datetime=None, end_datetime=None):
        images = [] 
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.ibw'):
                try:
                    afm_image = AFMImage(file_path)
                    
                    # Filter by datetime if specified
                    img_datetime = afm_image.get_datetime()
                    if img_datetime is not None:
                        if start_datetime and img_datetime < start_datetime:
                            continue
                        if end_datetime and img_datetime > end_datetime:
                            continue
                    
                    images.append(afm_image)
                except Exception as e: 
                    print(f"Error loading file {file_path}: {e}")
        
        images.sort(key=lambda image: image.get_datetime())
        return images

    def __len__(self): return len(self.images)
    def __getitem__(self, index): return self.images[index]
    def __iter__(self): yield from self.images

    def print_conversion_rates(self):
        for image in self.images:
            print(f"Image {image.bname} has the conversion rate of {image.get_conversion_rate()} microns per pixel")

    def review_maximum_Zpoints(self):
        rejected_images = []
        
        def on_click(event, image, max_position):
            # Check if the right mouse button was clicked (reject)
            if event.button == 3:  # Right-click
                rejected_images.append(image.get_filename())
                plt.close()  # Close the plot window
            # Check if the left mouse button was clicked (accept)
            elif event.button == 1:  # Left-click
                plt.close()  # Close the plot window
        
        for i, image in enumerate(self.images):
            max_value, max_position = image.get_maximum_Zpoint()
            if max_value is not None:
                # Display the image with a marker on the maximum point
                fig, ax = plt.subplots()
                ax.imshow(image.get_FlatZtrace(), cmap='gray', aspect='auto')
                ax.scatter(max_position[1], max_position[0], color='red', marker='x')  # Mark the max point
                ax.set_title(f"Image {i+1}: {image.get_datetime()} - Max Value: {max_value}")
                ax.set_xlabel('X Pixels')
                ax.set_ylabel('Y Pixels')

                # Connect the click event to the on_click function
                cid = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, image, max_position))

                # Show the plot
                plt.show()
        
        # Print the filenames of rejected images
        if rejected_images:
            print("Rejected images:")
            for filename in rejected_images:
                print(filename)
        else:
            print("No images were rejected.")

    def review_phase(self):
        rejected_images = []

        def on_click(event, image):
            if event.button == 3:
                rejected_images.append(image.get_filename())
                plt.close()
            elif event.button == 1: 
                plt.close() 

        for image in self.images:
            max_Zvalue, max_Zposition = image.get_maximum_Zpoint()
            max_Hvalue, max_Hposition = image.get_maximum_Hpoint()

            if max_Zvalue is not None and max_Hvalue is not None:
                phase_image = image.get_phase_retrace()
                file_name = image.get_filename() 
                binary_phase_image = np.where(phase_image >90,1,0) # all values greater than 90 are set to 1 

                fig, axes = plt.subplots(1, 2, figsize=(12,6))
                axes[0].imshow(phase_image, cmap='gray', aspect='auto')
                axes[0].scatter(max_Zposition[1], max_Zposition[0], color='red', marker='x')
                axes[0].scatter(max_Hposition[1], max_Hposition[0], color='blue')
                axes[0].set_title(f"Original Phase -{file_name}")

                axes[1].imshow(binary_phase_image, cmap='gray', aspect='auto')
                axes[1].scatter(max_Zposition[1], max_Zposition[0], color='red', marker='x')
                axes[1].scatter(max_Hposition[1], max_Hposition[0], color='blue')
                axes[1].set_title("Phase values less than 90 are set to 0")

                cid = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, image))
                mng = plt.get_current_fig_manager()
                try:
                    mng.window.state('zoomed')  # For TkAgg backend on Windows
                except AttributeError:
                 try:
                    mng.window.showMaximized()  # For Qt5Agg backend
                 except AttributeError:
                    try:
                        mng.frame.Maximize(True)  # For WxAgg backend
                    except AttributeError:
                        print("Maximization not supported on this backend.")
                plt.show()
        
        if rejected_images:
            print("Rejected images:")
            for filename in rejected_images:
                print(filename)
        else:
            print("No images were rejected.")

    def export_shift(self, length):
        data = {
            'File_Name' : [],
            'H_shift' : [],
            'Z_shift' : []
        }
        for image in self.images:
            data['File_Name'].append(image.get_filename())
            data['H_shift'].append(image.get_trimmed_trace_h(length)[3])
            data['Z_shift'].append(image.get_trimmed_trace_z(length)[3])
        df = pd.DataFrame(data)
        df.to_excel('Data_shift.xlsx', index=False)
        print("Data shift exported to Data_shift.xlsx")

    def export_deflection_time(self, length):
        data = {
            'File_Name' : [],
            'Time_Delta': [],
            'Max_deflection_h': [],
            'Max_deflection_z': []
        }
        t_0 = self[0].get_datetime()
        for image in self.images:
            data['File_Name'].append(image.get_filename())
            data['Max_deflection_h'].append(image.get_trimmed_trace_h(length)[2])
            data['Max_deflection_z'].append(image.get_trimmed_trace_z(length)[2])
            time_diff = image.get_datetime() - t_0
            data['Time_Delta'].append(time_diff.total_seconds())
        df = pd.DataFrame(data)
        df.to_excel("Time_V_deflection.xlsx", index=False)
        print("time V deflection exported to Time_V_deflection.xlsx")

    def navigate_images(self, initial_selections=None):
        """Open a small navigator window listing all images with their times,
        show which ones have selections (with checkmarks) and allow jumping
        back-and-forth. Stores selections in memory so returning to an image
        will show previous selections inside select_heights.
        """
        selections = initial_selections.copy() if initial_selections else {}

        root = tk.Tk()
        root.title('AFM Image Navigator')

        frame = ttk.Frame(root, padding=8)
        frame.pack(fill='both', expand=True)

        cols = ('#', 'Filename', 'Saved At', 'Done', 'Deflection (nm)', 'Offset (μm)')
        tree = ttk.Treeview(frame, columns=cols, show='headings', selectmode='browse')
        for c in cols:
            tree.heading(c, text=c)
        tree.column('#', width=30, anchor='center')
        tree.column('Filename', width=220)
        tree.column('Saved At', width=140)
        tree.column('Done', width=60, anchor='center')
        tree.column('Deflection (nm)', width=110, anchor='e')
        tree.column('Offset (μm)', width=130, anchor='e')

        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # populate
        for idx, img in enumerate(self.images):
            dt = img.get_datetime().strftime('%Y-%m-%d %H:%M:%S') if img.get_datetime() is not None else ''
            existing = selections.get(idx)
            done_flag = ''
            deflect = ''
            offset = ''
            if existing:
                slots = existing.get('selected_slots', [None, None])
                if any(slots):
                    done_flag = '✓'
                if slots[0] is not None and slots[1] is not None:
                    deflect = f"{slots[1][0] - slots[0][0]:.3f}"
            try:
                x_off = img.get_x_offset()
                y_off = img.get_y_offset()
            except (TypeError, ValueError):
                x_off = y_off = None
            if x_off is not None and y_off is not None:
                offset = f"{x_off:.3f}, {y_off:.3f}"
            tree.insert('', 'end', iid=str(idx), values=(idx+1, img.get_filename(), dt, done_flag, deflect, offset))

        def open_image(event=None):
            sel = tree.selection()
            if not sel:
                return
            idx = int(sel[0])
            img = self.images[idx]

            # Prepare initial slots if we have prior selections
            init_slots = None
            prev_time_offset = None
            if idx in selections:
                init_slots = selections[idx].get('selected_slots')
                prev_time_offset = selections[idx].get('time_offset')
                warning_msgs = selections[idx].get('warning_messages')
                if warning_msgs:
                    try:
                        messagebox.showwarning(
                            'Selection Warning',
                            '\n\n'.join(warning_msgs)
                        )
                    except Exception as exc:
                        print(f"Warning for image index {idx}: {warning_msgs} ({exc})")
                    selections[idx].pop('warning_messages', None)
            
            # Call existing select_heights; it now accepts initial selections and returns a dict
            res = vis.select_heights(img, initial_selected_slots=init_slots)
            if res is None:
                return
            if res.get('time_offset') is None and prev_time_offset is not None:
                res['time_offset'] = prev_time_offset
            
            # store result
            selections[idx] = res

            # Update tree: mark done and deflection (if two slots present compute difference)
            done = any(s is not None for s in res['selected_slots'])
            deflect = ''
            ss = res['selected_slots']
            if ss[0] is not None and ss[1] is not None:
                deflect = f"{ss[1][0] - ss[0][0]:.3f}"
            elif ss[0] is not None:
                deflect = f"{ss[0][0]:.3f}"
            tree.set(str(idx), 'Done', '✓' if done else '')
            tree.set(str(idx), 'Deflection (nm)', deflect)

        # Let the Treeview handle Up/Down navigation itself (default behavior).
        # Bind Enter and Space to open the selected image.
        tree.bind('<Double-1>', open_image)
        tree.bind('<Return>', lambda e: open_image())
        tree.bind('<space>', lambda e: open_image())

        # allow Delete key to clear selections for the selected image
        def delete_selection(event=None):
            sel = tree.selection()
            if not sel:
                return
            idx = int(sel[0])
            # remove stored selections (if any)
            if idx in selections:
                del selections[idx]
                print(f"Selections cleared for image index {idx}")
            # update UI to reflect cleared selections
            tree.set(str(idx), 'Done', '')
            tree.set(str(idx), 'Deflection (nm)', '')
        # Bind Delete to both the tree (when focused) and root (global for this window)
        tree.bind('<Delete>', delete_selection)
        root.bind('<Delete>', delete_selection)

        btn_frame = ttk.Frame(root, padding=6)
        btn_frame.pack(fill='x')
        btn_open = ttk.Button(btn_frame, text='Open Selected', command=open_image)
        btn_open.pack(side='left')

        def close():
            root.destroy()

        btn_close = ttk.Button(btn_frame, text='Close', command=close)
        btn_close.pack(side='right')

        root.mainloop()

        return selections
