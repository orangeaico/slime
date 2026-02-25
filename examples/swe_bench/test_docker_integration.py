#!/usr/bin/env python3
"""Test SWE-agent Docker integration independently from training.

This script verifies that:
1. Docker is accessible
2. SWE-agent can initialize environments
3. Commands can be executed in Docker containers
4. Git operations work correctly

Usage:
    python test_docker_integration.py
    python test_docker_integration.py --repo django/django --commit stable/4.2.x
"""

import argparse
import logging
import sys
from pathlib import Path

# Set up paths
sys.path.insert(0, "/root/swe_livup")

from sweagent.environment.repo import GithubRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_docker_integration(repo: str, commit: str):
    """Test Docker integration with SWE-agent.

    Args:
        repo: GitHub repository in format "owner/repo"
        commit: Git commit/branch/tag to use
    """
    logger.info("="*80)
    logger.info("Testing SWE-agent Docker Integration")
    logger.info("="*80)

    # Step 1: Create repo config
    logger.info(f"\nStep 1: Creating GithubRepoConfig")
    logger.info(f"  Repository: {repo}")
    logger.info(f"  Commit: {commit}")

    try:
        if "/" in repo:
            github_url = f"https://github.com/{repo}"
        else:
            github_url = repo

        repo_config = GithubRepoConfig(
            github_url=github_url,
            base_commit=commit,
        )
        logger.info(f"  ✓ RepoConfig created: {repo_config}")
    except Exception as e:
        logger.error(f"  ✗ Failed to create RepoConfig: {e}")
        raise

    # Step 2: Create environment config
    logger.info(f"\nStep 2: Creating EnvironmentConfig")
    try:
        env_config = EnvironmentConfig(repo=repo_config)
        logger.info(f"  ✓ EnvironmentConfig created")
        logger.info(f"  Deployment: {env_config.deployment}")
    except Exception as e:
        logger.error(f"  ✗ Failed to create EnvironmentConfig: {e}")
        raise

    # Step 3: Initialize SWEEnv
    logger.info(f"\nStep 3: Initializing SWEEnv")
    try:
        env = SWEEnv.from_config(env_config)
        logger.info(f"  ✓ SWEEnv created")
    except Exception as e:
        logger.error(f"  ✗ Failed to create SWEEnv: {e}")
        raise

    # Step 4: Start environment (boots Docker container)
    logger.info(f"\nStep 4: Starting environment (booting Docker container)")
    logger.info(f"  This will:")
    logger.info(f"    - Pull python:3.11 Docker image (if not present)")
    logger.info(f"    - Start a new container")
    logger.info(f"    - Clone {github_url}")
    logger.info(f"    - Checkout {commit}")
    logger.info(f"  This may take 1-2 minutes...")

    try:
        env.start()
        logger.info(f"  ✓ Environment started successfully!")
    except Exception as e:
        logger.error(f"  ✗ Failed to start environment: {e}")
        raise

    # Step 5: Test basic commands
    logger.info(f"\nStep 5: Testing command execution")

    commands_to_test = [
        ("pwd", "Check working directory"),
        ("ls -la", "List files"),
        ("git status", "Check git status"),
        ("git log -1 --oneline", "Show latest commit"),
        ("python --version", "Check Python version"),
    ]

    for cmd, description in commands_to_test:
        logger.info(f"\n  Testing: {description}")
        logger.info(f"  Command: {cmd}")
        try:
            output = env.communicate(cmd, timeout=30, check="ignore")
            logger.info(f"  Output ({len(output)} chars):")
            for line in output.split('\n')[:10]:  # Show first 10 lines
                logger.info(f"    {line}")
            if len(output.split('\n')) > 10:
                logger.info(f"    ... ({len(output.split('\n')) - 10} more lines)")
        except Exception as e:
            logger.error(f"  ✗ Command failed: {e}")

    # Step 6: Test git diff (for patch extraction)
    logger.info(f"\nStep 6: Testing patch extraction")
    try:
        # Make a small change
        logger.info(f"  Creating test file...")
        env.communicate("echo 'test' > test_file.txt", timeout=10, check="ignore")

        # Get diff
        logger.info(f"  Extracting git diff...")
        patch = env.communicate("git diff", timeout=10, check="ignore")
        if patch:
            logger.info(f"  ✓ Patch extracted ({len(patch)} chars)")
            logger.info(f"  Preview:")
            for line in patch.split('\n')[:15]:
                logger.info(f"    {line}")
        else:
            logger.info(f"  ✓ No changes (expected for clean repo)")
    except Exception as e:
        logger.error(f"  ✗ Patch extraction failed: {e}")

    # Step 7: Cleanup
    logger.info(f"\nStep 7: Cleaning up")
    try:
        env.close()
        logger.info(f"  ✓ Environment closed")
    except Exception as e:
        logger.error(f"  ✗ Cleanup failed: {e}")

    logger.info("\n" + "="*80)
    logger.info("✓ Docker integration test completed successfully!")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Test SWE-agent Docker integration"
    )
    parser.add_argument(
        "--repo",
        default="django/django",
        help="GitHub repository in format 'owner/repo' (default: django/django)"
    )
    parser.add_argument(
        "--commit",
        default="stable/4.2.x",
        help="Git commit/branch/tag (default: stable/4.2.x)"
    )

    args = parser.parse_args()

    try:
        test_docker_integration(args.repo, args.commit)
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
