from __future__ import annotations

from PySide6 import QtWidgets
from typing import Optional


class HelpDialog(QtWidgets.QDialog):
    """Help dialog providing comprehensive information about the GUI."""
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ExaFlow - Help")
        self.setModal(True)
        self.resize(800, 600)
        
        self._build_ui()
    
    def _build_ui(self) -> None:
        """Build the help dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create tabbed widget for different help sections
        tab_widget = QtWidgets.QTabWidget()
        
        # Overview tab
        overview_tab = self._create_overview_tab()
        tab_widget.addTab(overview_tab, "Overview")
        
        # Controls tab
        controls_tab = self._create_controls_tab()
        tab_widget.addTab(controls_tab, "Controls")
        
        # Visualization tab
        visualization_tab = self._create_visualization_tab()
        tab_widget.addTab(visualization_tab, "Visualization")
        
        # File formats tab
        file_formats_tab = self._create_file_formats_tab()
        tab_widget.addTab(file_formats_tab, "File Formats")
        
        # Troubleshooting tab
        troubleshooting_tab = self._create_troubleshooting_tab()
        tab_widget.addTab(troubleshooting_tab, "Troubleshooting")
        
        layout.addWidget(tab_widget)
        
        # Close button
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _create_overview_tab(self) -> QtWidgets.QWidget:
        """Create the overview tab content."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
        <h2>ExaFlow - Python GUI</h2>
        
        <p>This graphical user interface provides an intuitive way to run and visualize 3D Navier-Stokes simulations. 
        The GUI combines simulation execution capabilities with real-time 3D visualization using PyVista.</p>
        
        <h3>Key Features:</h3>
        <ul>
        <li><b>Simulation Management:</b> Run the configured case through the standard ExaFlow MPI command</li>
        <li><b>Real-time Visualization:</b> View simulation results in interactive 3D plots</li>
        <li><b>Auto-loading:</b> Automatically load the latest simulation results</li>
        <li><b>Multiple File Formats:</b> Support for VTK (.vtr) and CSV files</li>
        <li><b>Interactive Controls:</b> Toggle visualization elements and camera presets</li>
        </ul>
        
        <h3>Interface Layout:</h3>
        <ul>
        <li><b>Left Panel:</b> Simulation controls, file selection, and run log</li>
        <li><b>Right Panel:</b> 3D visualization viewer with toolbar controls</li>
        </ul>
        
        <h3>Getting Started:</h3>
        <ol>
        <li>Pick a case in "Preset", or open "Simulation Params..." and review the case</li>
        <li>Set the number of MPI processes (default: 4)</li>
        <li>Choose the output root</li>
        <li>Click "Run" to start <code>exaflow run</code> through MPI, or "Resume..." to continue from a checkpoint</li>
        <li>The viewer loads new result files from the output root</li>
        </ol>
        """)
        
        layout.addWidget(text)
        return widget
    
    def _create_controls_tab(self) -> QtWidgets.QWidget:
        """Create the controls tab content."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
        <h2>Simulation Controls</h2>
        
        <h3>Case Configuration:</h3>
        <ul>
        <li><b>Simulation Params...:</b> Edit the typed case that the run receives</li>
        <li><b>Preset:</b> Load one of the shipped cases; an edit in "Simulation Params..." sets it back to Custom</li>
        <li><b>MPI processes:</b> Number of parallel processes (1-512, default: 4)</li>
        <li><b>Output root:</b> Directory that receives one result directory per run</li>
        </ul>
        
        <h3>Auto-loading:</h3>
        <ul>
        <li><b>Auto-load newest:</b> Automatically loads the most recent .vtr or *_Total.csv file</li>
        <li>Updates every 2 seconds when enabled</li>
        <li>Helps monitor simulation progress in real-time</li>
        </ul>
        
        <h3>Run Controls:</h3>
        <ul>
        <li><b>Run:</b> Start the simulation with current settings</li>
        <li><b>Resume...:</b> Continue the run a checkpoint holds; the case comes from the checkpoint, not from the form</li>
        <li><b>Stop:</b> Terminate the running simulation</li>
        <li><b>Open File...:</b> Manually select a result file to load</li>
        <li><b>?:</b> Show this help dialog</li>
        </ul>
        
        <h3>Run Log:</h3>
        <ul>
        <li>Displays real-time output from the simulation</li>
        <li>Shows MPI process messages and simulation progress</li>
        <li>Automatically scrolls to show latest messages</li>
        <li>Limited to 5000 lines to prevent memory issues</li>
        </ul>
        """)
        
        layout.addWidget(text)
        return widget
    
    def _create_visualization_tab(self) -> QtWidgets.QWidget:
        """Create the visualization tab content."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
        <h2>3D Visualization Controls</h2>
        
        <h3>Display Options:</h3>
        <ul>
        <li><b>Axes:</b> Show coordinate axes orientation widget</li>
        <li><b>Outline:</b> Display domain boundary wireframe</li>
        <li><b>Cube Axes:</b> Show bounding box with axis labels</li>
        <li><b>Vectors:</b> Display velocity vector field as arrows</li>
        </ul>
        
        <h3>Vector Visualization:</h3>
        <ul>
        <li><b>Stride:</b> Controls vector sampling density (1-50, default: 4)</li>
        <li>Higher stride = fewer arrows = better performance</li>
        <li>Lower stride = more arrows = better detail</li>
        <li>Vectors are colored by pressure when available</li>
        </ul>
        
        <h3>Cross-section:</h3>
        <ul>
        <li><b>Slice:</b> Cut the result on one axis and show that plane by itself</li>
        <li><b>Axis:</b> The axis the plane cuts across (X, Y or Z)</li>
        <li><b>Position:</b> Moves the plane along that axis; the label states the coordinate and its unit</li>
        <li>The unit is metres for a .vtr file and cells for a CSV file, which carries no physical extent</li>
        <li>The camera faces the plane and stops rotating; turn Slice off to get the volume and free rotation back</li>
        <li>The control is disabled for a 1D or 2D result, which is already a cross-section</li>
        </ul>

        <h3>Camera Presets:</h3>
        <ul>
        <li><b>+X, -X:</b> View along positive/negative X-axis</li>
        <li><b>+Y, -Y:</b> View along positive/negative Y-axis</li>
        <li><b>+Z, -Z:</b> View along positive/negative Z-axis</li>
        <li><b>ISO:</b> Isometric view (45° angle)</li>
        </ul>
        
        <h3>Interactive Navigation:</h3>
        <ul>
        <li><b>Left mouse:</b> Rotate view</li>
        <li><b>Right mouse:</b> Pan view</li>
        <li><b>Mouse wheel:</b> Zoom in/out</li>
        <li><b>Middle mouse:</b> Pan view (alternative)</li>
        </ul>
        
        <h3>Color Mapping:</h3>
        <ul>
        <li>Uses "coolwarm" colormap (blue to red)</li>
        <li>Scalar bar shows data range and units</li>
        <li>Automatic formatting for scientific notation</li>
        <li>Pressure field typically displayed by default</li>
        </ul>
        """)
        
        layout.addWidget(text)
        return widget
    
    def _create_file_formats_tab(self) -> QtWidgets.QWidget:
        """Create the file formats tab content."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
        <h2>Supported File Formats</h2>
        
        <h3>VTK Rectilinear Grid (.vtr):</h3>
        <ul>
        <li><b>Format:</b> VTK XML Rectilinear Grid format</li>
        <li><b>Content:</b> 3D structured grid with scalar and vector fields</li>
        <li><b>Advantages:</b> High precision, supports multiple data arrays</li>
        <li><b>Use case:</b> Primary output format for detailed analysis</li>
        </ul>
        
        <h3>CSV Total Files (*_Total.csv):</h3>
        <ul>
        <li><b>Format:</b> Comma-separated values with spatial coordinates</li>
        <li><b>Content:</b> Point data including pressure and velocity</li>
        <li><b>Advantages:</b> Human-readable, easy to process</li>
        <li><b>Use case:</b> Quick visualization and data export</li>
        </ul>
        
        <h3>File Loading:</h3>
        <ul>
        <li><b>Automatic:</b> Auto-load feature monitors output directory</li>
        <li><b>Manual:</b> Use "Open File..." to select specific files</li>
        <li><b>Priority:</b> .vtr files preferred over CSV for visualization</li>
        <li><b>Error handling:</b> Unsupported files show warning messages</li>
        </ul>
        
        <h3>Data Fields:</h3>
        <ul>
        <li><b>Pressure:</b> Scalar field displayed as color mapping</li>
        <li><b>Velocity:</b> Vector field displayed as arrows</li>
        <li><b>Coordinates:</b> Spatial position (X, Y, Z)</li>
        <li><b>Custom fields:</b> Additional data arrays if present</li>
        </ul>
        """)
        
        layout.addWidget(text)
        return widget
    
    def _create_troubleshooting_tab(self) -> QtWidgets.QWidget:
        """Create the troubleshooting tab content."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setHtml("""
        <h2>Troubleshooting Guide</h2>
        
        <h3>Common Issues:</h3>
        
        <h4>Viewer Unavailable:</h4>
        <ul>
        <li><b>Symptom:</b> "Viewer unavailable. Install 'pyvista' and 'pyvistaqt'"</li>
        <li><b>Solution:</b> Install required packages: <code>pip install pyvista pyvistaqt</code></li>
        <li><b>Note:</b> Restart the GUI after installation</li>
        </ul>
        
        <h4>Simulation Won't Start:</h4>
        <ul>
        <li><b>Check:</b> The case values in "Simulation Params..." are valid</li>
        <li><b>Check:</b> MPI is installed and accessible</li>
        <li><b>Check:</b> The ExaFlow command is available in the active Python environment</li>
        </ul>
        
        <h4>No Results Loading:</h4>
        <ul>
        <li><b>Check:</b> Output directory path is correct</li>
        <li><b>Check:</b> Simulation is actually producing files</li>
        <li><b>Check:</b> File naming matches expected patterns (*_Total.csv, *.vtr)</li>
        <li><b>Try:</b> Manual file loading with "Open File..."</li>
        </ul>
        
        <h4>Performance Issues:</h4>
        <ul>
        <li><b>Large datasets:</b> Increase vector stride to reduce arrow count</li>
        <li><b>Slow rendering:</b> Disable unnecessary overlays (axes, outline)</li>
        <li><b>Memory issues:</b> Close and reopen large files</li>
        <li><b>MPI processes:</b> Adjust number based on available cores</li>
        </ul>
        
        <h3>System Requirements:</h3>
        <ul>
        <li><b>Python:</b> 3.8 or higher</li>
        <li><b>PySide6:</b> For GUI framework</li>
        <li><b>PyVista:</b> For 3D visualization</li>
        <li><b>PyVistaQt:</b> For Qt integration</li>
        <li><b>MPI:</b> For parallel simulation execution</li>
        <li><b>NumPy:</b> For numerical computations</li>
        </ul>
        
        <h3>Getting Help:</h3>
        <ul>
        <li>Check the run log for detailed error messages</li>
        <li>Verify all file paths and permissions</li>
        <li>Test the case with <code>mpiexec -n 1 exaflow run --case case.xml</code></li>
        <li>Ensure all dependencies are properly installed</li>
        </ul>
        """)
        
        layout.addWidget(text)
        return widget
