#!/usr/bin/env python3
"""
Convert a GIF file to a folder containing individual PNG frames.

Usage:
    python gif_to_frames.py <gif_file_path>
    
Example:
    python gif_to_frames.py Animation.gif
"""

import os
import sys
from PIL import Image
import argparse


def gif_to_frames(gif_path, output_dir=None):
    """
    Convert a GIF file to individual PNG frames.
    
    Args:
        gif_path (str): Path to the input GIF file
        output_dir (str, optional): Output directory. If None, uses GIF filename without extension
    
    Returns:
        str: Path to the output directory containing the frames
    """
    # Validate input file
    if not os.path.exists(gif_path):
        raise FileNotFoundError(f"GIF file not found: {gif_path}")
    
    if not gif_path.lower().endswith(('.gif', '.GIF')):
        raise ValueError("Input file must be a GIF file")
    
    # Set output directory
    if output_dir is None:
        gif_name = os.path.splitext(os.path.basename(gif_path))[0]
        output_dir = os.path.join(os.path.dirname(gif_path), gif_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Open the GIF file
        with Image.open(gif_path) as gif:
            print(f"Processing GIF: {gif_path}")
            print(f"GIF info: {gif.size[0]}x{gif.size[1]} pixels, {gif.n_frames} frames")
            
            # Convert each frame to PNG
            for frame_num in range(gif.n_frames):
                # Seek to the specific frame
                gif.seek(frame_num)
                
                # Convert to RGB if necessary (handles palette mode)
                if gif.mode in ('P', 'RGBA'):
                    frame = gif.convert('RGBA')
                else:
                    frame = gif.convert('RGB')
                
                # Generate output filename
                frame_filename = f"frame_{frame_num + 1:02d}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # Save the frame as PNG
                frame.save(frame_path, 'PNG')
                print(f"Saved frame {frame_num + 1}/{gif.n_frames}: {frame_filename}")
            
            print(f"\nConversion complete! {gif.n_frames} frames saved to: {output_dir}")
            return output_dir
            
    except Exception as e:
        print(f"Error processing GIF: {e}")
        raise


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a GIF file to individual PNG frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gif_to_frames.py Animation.gif
  python gif_to_frames.py /path/to/my_animation.gif
  python gif_to_frames.py Animation.gif --output /custom/output/dir
        """
    )
    
    parser.add_argument('gif_file', help='Path to the input GIF file')
    parser.add_argument('-o', '--output', help='Output directory (default: same as GIF filename)')
    
    args = parser.parse_args()
    
    try:
        output_dir = gif_to_frames(args.gif_file, args.output)
        print(f"\nSuccess! Frames are available in: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
