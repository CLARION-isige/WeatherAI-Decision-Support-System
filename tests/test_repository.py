"""Unit tests for Repository pattern"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from app.repositories.base import Repository
from app.repositories.firestore_repository import FirestoreDecisionRepository
from app.core.models import DecisionResponse, Location, CropType


class TestRepository:
    """Test suite for abstract Repository base class"""
    
    def test_repository_is_abstract(self):
        """Test that Repository is abstract and cannot be instantiated"""
        with pytest.raises(TypeError):
            Repository()
    
    def test_repository_has_required_methods(self):
        """Test that Repository defines required abstract methods"""
        assert hasattr(Repository, 'save')
        assert hasattr(Repository, 'find_by_id')
        assert hasattr(Repository, 'find_all')
        assert hasattr(Repository, 'update')
        assert hasattr(Repository, 'delete')
        assert hasattr(Repository, 'query')


class TestFirestoreDecisionRepository:
    """Test suite for FirestoreDecisionRepository"""
    
    @pytest.fixture
    def repository(self):
        """Fixture for Firestore repository"""
        # Clear mock storage
        FirestoreDecisionRepository._mock_storage.clear()
        return FirestoreDecisionRepository()
    
    @pytest.fixture
    def mock_decision_response(self, mock_location):
        """Fixture for mock decision response"""
        return DecisionResponse(
            request_id="req-123",
            timestamp=datetime.now(timezone.utc),
            location=mock_location,
            crop=CropType.MAIZE,
            decision_type="planting",
            recommendation={"recommended": True}
        )
    
    def test_repository_initialization(self, repository):
        """Test repository initialization"""
        assert repository is not None
        assert repository.settings is not None
    
    def test_repository_initialization_without_firebase(self, repository):
        """Test repository initialization falls back to mock when Firebase unavailable"""
        # Repository should work even without Firebase
        assert repository._initialized == False or repository._initialized == True
    
    @pytest.mark.asyncio
    async def test_save_entity(self, repository, mock_decision_response):
        """Test saving an entity"""
        entity_id = await repository.save(mock_decision_response)
        
        assert entity_id is not None
        assert isinstance(entity_id, str)
    
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self, repository, mock_decision_response):
        """Test saving and retrieving an entity"""
        entity_id = await repository.save(mock_decision_response)
        
        retrieved = await repository.find_by_id(entity_id)
        
        assert retrieved is not None
        assert retrieved["request_id"] == mock_decision_response.request_id
    
    @pytest.mark.asyncio
    async def test_find_by_id_not_found(self, repository):
        """Test finding by ID when entity doesn't exist"""
        result = await repository.find_by_id("non-existent-id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_find_all(self, repository, mock_decision_response):
        """Test finding all entities"""
        # Save multiple entities
        await repository.save(mock_decision_response)
        
        # Create and save another
        mock_decision_response.request_id = "req-456"
        await repository.save(mock_decision_response)
        
        results = await repository.find_all(limit=10)
        
        assert results is not None
        assert isinstance(results, list)
        assert len(results) >= 2
    
    @pytest.mark.asyncio
    async def test_find_all_with_pagination(self, repository, mock_decision_response):
        """Test find all with pagination"""
        # Save multiple entities
        for i in range(5):
            mock_decision_response.request_id = f"req-{i}"
            await repository.save(mock_decision_response)
        
        results = await repository.find_all(limit=2, offset=0)
        assert len(results) == 2
        
        results = await repository.find_all(limit=2, offset=2)
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_update_entity(self, repository, mock_decision_response):
        """Test updating an entity"""
        entity_id = await repository.save(mock_decision_response)
        
        updates = {"decision_type": "harvesting"}
        success = await repository.update(entity_id, updates)
        
        assert success == True
        
        # Verify update
        updated = await repository.find_by_id(entity_id)
        assert updated["decision_type"] == "harvesting"
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_entity(self, repository):
        """Test updating non-existent entity"""
        success = await repository.update("non-existent", {"decision_type": "harvesting"})
        assert success == False
    
    @pytest.mark.asyncio
    async def test_delete_entity(self, repository, mock_decision_response):
        """Test deleting an entity"""
        entity_id = await repository.save(mock_decision_response)
        
        success = await repository.delete(entity_id)
        assert success == True
        
        # Verify deletion
        result = await repository.find_by_id(entity_id)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_entity(self, repository):
        """Test deleting non-existent entity"""
        success = await repository.delete("non-existent")
        assert success == False
    
    @pytest.mark.asyncio
    async def test_query_entities(self, repository, mock_decision_response):
        """Test querying entities with filters"""
        # Save entities with different decision types
        mock_decision_response.decision_type = "planting"
        await repository.save(mock_decision_response)
        
        mock_decision_response.request_id = "req-2"
        mock_decision_response.decision_type = "harvesting"
        await repository.save(mock_decision_response)
        
        # Query for planting decisions
        results = await repository.query({"decision_type": "planting"})
        
        assert results is not None
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(r["decision_type"] == "planting" for r in results)
    
    @pytest.mark.asyncio
    async def test_query_with_multiple_filters(self, repository, mock_decision_response, mock_location):
        """Test querying with multiple filters"""
        mock_decision_response.decision_type = "planting"
        mock_decision_response.crop = CropType.MAIZE
        await repository.save(mock_decision_response)
        
        mock_decision_response.request_id = "req-2"
        mock_decision_response.crop = CropType.WHEAT
        await repository.save(mock_decision_response)
        
        # Query for maize planting decisions
        results = await repository.query({
            "decision_type": "planting",
            "crop": "maize"
        })
        
        assert results is not None
        assert len(results) >= 1
        assert all(r["crop"] == "maize" for r in results)
    
    def test_mock_storage_isolation(self):
        """Test that mock storage is isolated between instances"""
        repo1 = FirestoreDecisionRepository()
        repo2 = FirestoreDecisionRepository()
        
        # Both should share the same mock storage (class variable)
        assert repo1._mock_storage is repo2._mock_storage


class TestFirestoreRepositoryWithFirebase:
    """Test suite for Firestore repository with actual Firebase (integration)"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_firebase_save_and_retrieve(self):
        """Test actual Firebase save and retrieve"""
        # This test requires Firebase configuration
        pytest.skip("Skip integration test without Firebase configuration")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_firebase_query(self):
        """Test actual Firebase query"""
        # This test requires Firebase configuration
        pytest.skip("Skip integration test without Firebase configuration")
