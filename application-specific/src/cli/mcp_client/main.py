import asyncio
from pathlib import Path
import os
import httpx

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Use a long SSE read timeout so long-running tools (e.g., training) don't trigger client timeouts
transport = StreamableHttpTransport(
    url="http://127.0.0.1:8000/mcp",
    sse_read_timeout=60 * 60 * 6,  # 6 hours
)
client = Client(transport)


async def main():
    async with client:
        # Basic server interaction & session
        await client.ping()

        # Create a session on the server
        init_res = await client.call_tool("initialize_session")
        session_sc = getattr(init_res, "structured_content", None)
        session_id = session_sc.get("session_id") if isinstance(session_sc, dict) else None
        print(f"Initialized session: {session_id}")

        # Use the provided example architecture file
        project_dir = Path.cwd() / "hohooo"
        # arch_path = project_dir / "architectures/custom_architecture_architecture.json"
        # if not arch_path.exists():
        #     print(f"Architecture file not found: {arch_path}")
        #     await client.call_tool("close_session")
        #     return

        # print(f"Using architecture file: {arch_path}")

        # # Call generate_pytorch tool and generate into the existing my_first_project directory
        # gen_output_dir = project_dir
        # print("Calling generate_pytorch into:", gen_output_dir)
        # gen_res = await client.call_tool(
        #     "generate_pytorch",
        #     {
        #         "architecture_file": arch_path,
        #         "output_dir": gen_output_dir,
        #         "model_name": "GeneratedFromClient",
        #         "include_requirements": False,
        #     }
        # )

        # gen_content = gen_res.structured_content or {}
        # print("generate_pytorch response:", gen_content)

        # if gen_content.get("status") == "error":
        #     print(gen_content.get("message"))
        #     await client.call_tool("close_session")
        #     return

        # # Determine project_output from generator response (align keys with server)
        # details = gen_content.get("details", {})
        # # Server returns either 'output_path' (per template generators) or 'output_dir' (generate_pytorch wrapper)
        # project_output = details.get("output_path") or details.get("output_dir")
        # if not project_output and details.get("main_py_path"):
        #     # derive project root from returned main_py_path: main_py_path = <output_path>/Pretrained_Output/main.py
        #     main_py = Path(details.get("main_py_path"))
        #     # project root is two parents up from main.py
        #     project_output = str(main_py.parent.parent)

        # if not project_output:
        #     # final fallback: use the gen_output_dir itself
        #     project_output = str(gen_output_dir.resolve())

        # print(f"Project output directory: {project_output}")

        # Call test_image on the server. Fill the Path values below before running.
        image_path = Path(r"C:\Users\Ahmed hosam\Desktop\grad\Application-Specific-Deep-Learning-Accelerator-Designer\BCCD.v3-raw.coco_yolox\val2017\BloodImage_00004_jpg.rf.5abe41b92c2d446545da27876795e4ec.jpg")  # TODO: set image file path (e.g. r"C:/path/to/image.jpg")
        model_path = Path(r"C:\Users\Ahmed hosam\Desktop\grad\Application-Specific-Deep-Learning-Accelerator-Designer\rika\YOLOX\YOLOX_outputs\yolox_base\best_ckpt.pth")  # TODO: set YOLOX model weights .pth file
        hgd_ckpt_path = Path(r"C:\Users\Ahmed hosam\Desktop\grad\Application-Specific-Deep-Learning-Accelerator-Designer\weights\Defense.pt")  # TODO: set HGD checkpoint .pt file
        class_names_path = Path(r"C:\Users\Ahmed hosam\Desktop\grad\Application-Specific-Deep-Learning-Accelerator-Designer\BCCD.v3-raw.coco_yolox\classes.txt")  # TODO: set class names file (one name per line)

        print("Calling test_image...")
        try:
            test_res = await client.call_tool(
                "test_image",
                {
                    "image_path": image_path,
                    "model_path": model_path,
                    "hgd_ckpt_path": hgd_ckpt_path,
                    "class_names_path": class_names_path,
                    "output_dir": Path.cwd() / "test_results",
                }
            )
            print("test_image response:", test_res.structured_content)
        except httpx.ReadTimeout:
            # If the SSE stream times out, the server may still be processing.
            print("Warning: test_image stream timed out while waiting for server events.\n"
                  "The server may still be processing; check server logs and outputs.")

        # Close session
        close_res = await client.call_tool("close_session")
        print("Closed session:", close_res.structured_content)


if __name__ == "__main__":
    asyncio.run(main())
